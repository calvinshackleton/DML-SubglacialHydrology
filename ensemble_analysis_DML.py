#-----------------------------------------------------------------------------
"""
Ensemble analysis of sequential gaussian simulated grid output

Created on 31-03-2023
@author: calvin.shackleton@npolar.no
"""
#%%-----------------------------------------------------------------------------
import os, glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from fprint import fprint
# from osgeo import gdal
# import gdal
import rasterio as rio
from rasterio.crs import CRS
import pandas as pd
import create_figs_DML as cfig
from tqdm import tqdm
import pyproj
import sys
sys.path.append("C:\\Users\\calvin.shackleton\\Dropbox\\Scripts\\Utility")
os.environ['PROJ_LIB'] = 'C:\\Users\\calvin.shackleton\\.conda\\envs\\geospatial\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\calvin.shackleton\\.conda\\envs\\geospatial\\Library\\share'
    
#%%-----------------------------------------------------------------------------
region= [-700000,1300000,1800000,2230000] # Custom CRS full region 
target_crs= pyproj.CRS(CRS.from_dict({'proj':"stere",'lat_ts': -71,'units': 'm',
                                      'lat_0':-90, 'lon_0':10, 'ellps': 'WGS84'}))
drive= 'S:/'

res=    1000       # Out grid resolution (m)

vgram_version=  'v2.5'
result_version= 'v3.4'
out_version=    'v1.0'

indir= f'{drive}DronningMaudLand - Topographic uncertainty/'
variogram_dir= f'{indir}variograms/DML_{vgram_version}/'
folder= f'{indir}_results_/DML_{result_version}/'
ftag= 'k_1.1_hydp'


outdir= f'{indir}/ensemble_analysis/hydrology_{out_version}/{ftag}/' # 
if not os.path.exists(outdir):
    os.makedirs(outdir)

#---Options True/False------------------
calc_stream_probability= True
topo_ensemble_analysis= False
plot_regional_histograms= False


#%%-----------------------------------------------------------------------------
def count_raster(inraster_path,in_counter,no_data_val=[-9999]):
    with rio.open(inraster_path) as src:    # Read lakes raster + metadata
        inraster= src.read(1,masked=False)
        global raster_meta
        raster_meta= src.profile
        
        if type(in_counter)==int:           # Use lakes raster as first count
            in_counter= inraster.copy()
            in_counter[inraster!= no_data_val]=1
            in_counter[inraster== no_data_val]=0
        else:                               # use subsequent rasters to add to count
            inraster[inraster!= no_data_val]=1
            inraster[inraster== no_data_val]=0
            in_counter = in_counter + inraster
        return in_counter
    
#-----------------------------------------------------------------------------
def add_raster(inraster_path,in_total,i):
    with rio.open(inraster_path) as src:    # Read lakes raster + metadata
        inraster= src.read(1,masked=False)
        global raster_meta
        raster_meta= src.profile
        
        if type(in_total)==int:           # Use lakes raster as first count
            in_total= inraster.copy()
        else:                             # use subsequent rasters to add to count
            in_total = in_total + inraster
        i+=1
        return in_total,i
    
#-----------------------------------------------------------------------------
def write_raster(inraster,path,name,raster_meta,crs=target_crs):
    raster= rio.open(path+name,   # write .tiff
                     'w', driver='GTiff',
                     height = raster_meta['height'],
                     width = raster_meta['width'],
                     count=1, dtype=str(inraster.dtype),
                     crs= target_crs,
                     transform= raster_meta['transform']) 
    raster.write(inraster,1)
    raster.close()
    raster= None
    
#%% Raster stack analysis ---------------------------------------------------
def raster_stack_analysis(folder,outdir,outname,search='*/*.tif'):
    import scipy.stats
    from scipy.stats import median_abs_deviation as mads
    import rasterio as rio
    
    glist= [f for f in glob.glob(os.path.join(folder,(search)))]
    # print(glist)
    
    with rio.open(glist[0]) as src0:
        meta = src0.meta
    meta.update(count = len(glist)) # update meta to reflect no. of layers
    
    with rio.open(f'{outdir}{outname}_stack.tif', 'w', **meta) as dst:
        for id, layer in enumerate(glist, start=1):
            with rio.open(layer) as src1:
                dst.write_band(id, src1.read(1))            
    
    with rio.open(f'{outdir}{outname}_stack.tif') as stack_src:
        stack_data = stack_src.read(masked=True)
        stack_data[stack_data==-9999] = 0
        stack_meta = stack_src.profile
    
        # Calculate statistics for the raster stack----------
        std= np.std(stack_data, axis=0)
        mean= np.mean(stack_data, axis=0)
        median= np.median(stack_data, axis=0)
        mode= scipy.stats.mode(stack_data, axis=0) #returns mode and counts [0][1]
        mode= mode[0].squeeze()
        mads= mads(stack_data, axis=0)
        total= np.sum(stack_data, axis=0)
        maximum= stack_data.max(axis=0)
        minimum= stack_data.min(axis=0)
    
        # write out results and plot results using dictionary mapping
        func_map= {'std':std,
                   'mean':mean,
                   'median':median,
                   'mode':mode, 
                   'mads':mads,
                   'sum':total,
                   'max':maximum,
                   'min':minimum,
                   }
    
        for key, value in func_map.items():
            write_raster(value,outdir,f'{outname}_ensemble_{key}.tif',meta,target_crs)
        
#-----------------------------------------------------------------------------
def create_grid(grid_template_path):
    import geopandas as gpd
    from shapely.geometry import box, Polygon
    g_temp= rio.open(grid_template_path)
    
    min_x, min_y, max_x, max_y = g_temp.bounds # Get extent of template grid
    region= [min_x,max_x,min_y,max_y]          # output region also
    side_length= g_temp.res[0]                 # grid cell size 
    
    cells_list = []

    # Iterate list of x (then y) values that define column (then row) positions 
    for x in np.arange(min_x, max_x, side_length):
        for y in np.arange(min_y, max_y, side_length):
            cells_list.append(
                box(x, y, x+side_length, y+side_length))  # Create a box + append 

    # Create grid from list of cells
    grid= gpd.GeoDataFrame(cells_list, columns= ['geometry'], crs= target_crs)
    grid["Grid_ID"] = np.arange(len(grid)) # Column that assigns each grid a number

    return grid, region

#%%-----------------------------------------------------------------------------
def regional_histograms(raster_path,outdir,target_crs,buffer_size=200000):
    import rasterstats as rs
    import rasterio as rio
    from rasterio import mask
    import rioxarray as riox
    import geopandas as gpd
    from shapely.geometry import mapping
    import matplotlib.ticker as plticker
    
    # raster_path= "S:/DML/ensemble_analysis/1.0/Bed_ensemble_mads.tif"
    GL_shp= 'S:/QGIS/Quantarctica3/QA-RINGS/GroundingLine/RINGS_workshop_grounded_2022.shp'
    
    # Create a clip of MAD.tif using grounding line buffer
    GL_shp= gpd.read_file(GL_shp)
    groundingline= gpd.GeoDataFrame.explode(GL_shp,index_parts=True).boundary.buffer(50000)
    groundingline= groundingline.to_crs(target_crs)
    
    with rio.open(raster_path) as src:
        grid, transform= mask.mask(src,groundingline,
                                   nodata=np.nan, crop=True)  # crop=True,
        with rio.open(outdir+'MAD_near_GL.tif', 'w',
                      **{'driver': 'GTiff',
                        'dtype': rio.float32,
                        'nodata': None,
                        'width': grid.squeeze().shape[1],
                        'height': grid.squeeze().shape[0],
                        'count': 1,
                        'crs': target_crs,  # Replace with your CRS
                        'transform': transform
                        }) as dst:
            dst.write(grid.squeeze(), 1)
        
    cfig.topo_figure(outdir+'MAD_near_GL.tif',
                     outdir+'MAD_near_GL.png',
                     'Ensemble MAD (m)', setminmax=[0,200], contours=True,
                     cmap='magma',**{'stations':True,'ground_line':True})
    
    station_list= pd.read_csv(r'S:\DronningMaudLand - Topographic uncertainty\other\stations.csv')
    project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), 
                                          target_crs, always_xy=True)
    station_list['x'], station_list['y'] = project.transform(
        station_list.Longitude, station_list.Latitude)

    vector= gpd.GeoDataFrame(data= station_list, crs= target_crs,
                            geometry= gpd.points_from_xy(station_list.x, station_list.y)) 
    vector= vector.buffer(buffer_size)
    station_list['polygon']= vector
    
    #--------------------------------------------------
    # Loop through and extract stats individually
    #--------------------------------------------------
    h_fig= plt.figure(figsize=(5,5))
    h_ax= plt.axes()
    stats_table= pd.DataFrame()
    
    for row in station_list.iterrows():
        
        stats= rs.zonal_stats(row[1].polygon, raster_path, stats=rs.utils.VALID_STATS).pop()
        stats= pd.DataFrame(columns=[row[1].Station], data=stats.values(),
                            index=pd.Series(stats.keys()))
        stats_table= pd.concat([stats_table, stats],axis=1)
        
        with rio.open(outdir+'MAD_near_GL.tif') as src:
            grid, subset_transform= mask.mask(src,[row[1].polygon],crop=True,
                                      nodata=np.nan)
            with rio.open(f'{outdir}MAD_near_{row[1].Station}.tif', 'w',
                      **{'driver': 'GTiff',
                        'dtype': rio.float32,
                        'nodata': None,
                        'width': grid.squeeze().shape[1],
                        'height': grid.squeeze().shape[0],
                        'count': 1,
                        'crs': target_crs,  # Replace with your CRS
                        'transform': subset_transform
                        }) as dst:
                dst.write(grid.squeeze(), 1)
            
            cfig.topo_figure(f'{outdir}MAD_near_{row[1].Station}.tif',
                             f'{outdir}MAD_near_{row[1].Station}.png',
                             'Ensemble MAD (m)', setminmax=[0,200],
                             cmap='magma',**{'stations':True,'ground_line':True})
            
        d=  np.hstack(grid.squeeze())
        df= pd.DataFrame(d, columns=['data'])
        df= df.dropna(how='all')
        df.reset_index()
        
        # Plot histogram from extracted values
        plt.sca(h_ax)
        plt.hist(df, density=False,   # density vs frequency
                 bins='auto',      # auto | sqrt | fd
                 histtype= 'step',
                 label=row[1].Station,        # string or sequence of strings
                     )
    plt.xlabel('MAD (m)'); plt.ylabel('Count');
    plt.legend()

    # Add the grid
    h_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=50))
    h_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=200))
    h_ax.grid(which='both', axis='both', linestyle='--',
              linewidth=0.4, alpha=0.5)
    
    # Save figure
    h_fig.savefig(f'{outdir}MAD_DML.png',
                format='png',
                transparent=True,
                bbox_inches="tight",
                dpi= 600)
    plt.close()
    
    stats_table.to_csv(f'{outdir}regional_statistics.csv',index=True)
    
    
#%%-------------------------------------------------------------------------
#----Stream probability---------------------------------------------------
#-------Stream stats workflow---------------------------------------------
if bool(calc_stream_probability):
    # Create raster stack and output statistics for ensemble
    raster_stack_analysis(folder,outdir,'strahler_streams',
                          search=f'*/{ftag}/strahler_streams.tif')  
    
    # Plot results, dict for colormaps
    func_map= {'std':'magma',
                'mean':'magma',
                'median':'magma',
                # 'mode':'magma',
                'mads':'magma',
                # 'sum':'magma',
                # 'max':'magma',
                # 'min':'magma',
                }
    for key, value in func_map.items():
        cfig.topo_figure(f'{outdir}strahler_streams_ensemble_{key}.tif',
                        f'{outdir}ensemble_streams_{key}.png',
                        cbar_label= f'{key}', cmap= value,
                        # cmap_min_max=['min','mid','max'],
                        histogram=False)
        fprint(f'Plotted: {key}')
        
        
    #------Stream counter method------------------------------------------
    in_counter= 0  # assign empty variable for first loop
    for inpath in sorted(glob.glob(os.path.join(folder,('*')))):
        inraster_path= f'{inpath}/{ftag}/strahler_streams.tif'
        # print(inraster_path)
        
        in_counter= count_raster(inraster_path,in_counter,no_data_val=[0]) # run counter
        fprint(f'Added from: {inraster_path}')
    
    streams_probability= in_counter/in_counter.max() # calculate lake probability 
    write_raster(streams_probability,outdir,'streams_probability.tif',raster_meta)

    cfig.topo_figure(outdir+'streams_probability.tif',
                     outdir+'Stream_prob_map.png',
                     'Stream probability',contours=True,cmap='PuBu',setminmax=[0.01,0.7],
                     **{'ice_drain':True,'ground_line':True,'stations':True}) # cm.ice_r | PuBu
    
    
#%%-------------------------------------------------------------------------
#----Topography ensemble analysis-----------------------------------------
#-------------------------------------------------------------------------
if bool(topo_ensemble_analysis):
    import create_figs_DML as cfig
    # Create raster stack and output statistics for ensemble
    raster_stack_analysis(folder,outdir,'Bed',
                          search='*/SGSim.tif')  
    
    # Plot results, dict for colormaps
    func_map= {'std':['magma',[0,500]],        # [0,260]
               'mean':['viridis',False],
               'median':['viridis',False],
               'mode':['viridis',False],
               'mads':['magma',[0,200]],
               }
    for key, value in func_map.items():
        cfig.topo_figure(f'{outdir}bed_ensemble_{key}.tif',
                     f'{outdir}bed_ensemble_{key}.png', 
                     f'ensemble {key} (m)',
                     histpath= f'{outdir}bed_ensemble_{key}_hist.png', 
                     setminmax=value[1],contours=False, histogram=True,
                     cmap=value[0], **{'background':False,'stations':True,
                         'ice_drain':False,'ground_line':False})
        
    if bool(plot_regional_histograms):
        regional_histograms(f'{outdir}Bed_ensemble_mads.tif',
                            outdir,target_crs,buffer_size=200000)
        
#%%-------------------------------------------------------------------------