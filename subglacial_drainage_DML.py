#------------------------------------------------------------------------------
"""
Calculate hydraulic potential surfaces using ice thickness and bed elevation
    - Most of workflow in python, but water routing tools use Matlabs TopoToolBox
    - Need to precondition the DEM to remove small sinks, e.g. below certain depth or less than n pixels
    - Set lake fill level and minupstream area 

Created on Tue Dec 14 17:11:31 2021
@author: calvin.shackleton@npolar.no
"""
#------------------------------------------------------------------------------
from osgeo import gdal
import rioxarray as rxr
import xarray as xr
import matlab.engine 
import numpy as np
import os, shutil, glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import create_figs_DML as cfig
import geopandas as gpd
from tqdm import tqdm
import pyproj
from rasterio.mask import mask
from rasterio.crs import CRS
import rasterio as rio
import sys
import workdir,fprint

run_in_parallel= False
imin= 0
imax= 50
in_parallel_max= 10

plot_obs_lakes = False # plot observed lakes? matlab: 1 python: True
lake_shapefiles = False
saveRasters= True

dir_id= 'SubglacialDrainage'   

#-----------------------------------------------------------------------------
def hydrology(indir,bed,h_ice,isurf,spatial_ref=None,tmp_mask=False):
#-----------------------------------------------------------------------------
# Variables and constants, save to text (read into matlab)--------------------
    k= 1     # cryostatic pressure (water pressure/ice overburden)
    pIce= 917       # ice density (kg m-3)
    pWater= 1000    # water density (kg m-3)
    gravity= 9.81   # acceleration due to gravity (m s-1)
    
    Qfile= f'{indir}hyd_p.tif'  # h_head|hyd_p (input surface)
    min_lake_level= 1    # Minimum lake depth
    max_lake_level= 500  # Maximum lake depth (mean depth 200m in Siegert inventory)
    min_lake_area=  1    # Minimum lake area in pixels (lake raster>polygon tool) 
    upstream_area=  1500  # Upstream pixels to init. stream 
    elevate_minima_loops= 5 # Num. times to run fill tools 
    max_elev_min_area=    15 # Max fill area (pixels) 
    topo_exagg= 5       # topo exaggeration for map output 15
    
    #--- Save params ---------------------------------------
    if not os.path.exists(f'{indir}'):
        os.makedirs(f'{indir}')
    print("variable    value", 
    "\ndirectory   ", f'"{indir}"',
    "\nfile   ", f'"{Qfile}"',
    "\nk   ", k,
    "\npIce   ", pIce,
    "\npWater   ", pWater,
    "\ngravity   ", gravity,
    "\nmin_lake_level   ", min_lake_level,
    "\nmax_lake_level   ", max_lake_level,
    "\nmin_lake_area   ", min_lake_area,
    "\nupstream_area   ", upstream_area,
    # "\nplot_obs_lakes   ", plot_obs_lakes,
    "\nelevate_minima_loops   ", elevate_minima_loops,
    "\nmax_elev_min_area   ", max_elev_min_area,
    "\ntopo_exagg   ", topo_exagg,
    file= open(f'{indir}hydrology_params.txt','w'))
    with open(f'{indir}hydrology_params.txt','r') as f:
        params= f.read(); print(params)
    

    # Hydraulic head/potential calculation-------------------------------------
    Q= (pWater*gravity*bed) + (k*(pIce*gravity*h_ice)) # hyd. pressure potential
    Q= Q.round().astype(int)
    # Q.rio.write_crs('epsg:3031',inplace=True)

    # if tmp_mask:
    #     Q= Q.where(bed_above_pmp>=0)      # mask results below pmp
    
    Q = xr.where(Q <= -9999, np.nan, Q)   # set to 0 or np.nan?
    Q['spatial_ref']= spatial_ref
    
    Q_file = f'{indir}hyd_p.tif'
    Q.rio.to_raster(Q_file,driver='GTiff', nodata=np.nan)   # write output to tiff  |nodata=-9999|
    cfig.topo_figure(f'{Q_file}',f'{indir}/hyd_p.png',
                        f'kpa',cmap='viridis')
    
            
#------------------------------------------------------------------------------
# Run matlab water routing tools in python using matlab engine
#------------------------------------------------------------------------------
    eng= matlab.engine.start_matlab()
    eng.addpath('C:/Users/calvin.shackleton/Dropbox/Scripts/Matlab/water_routing_DML')
    eng.Subglacial_hydrology_DML(indir,nargout=0) # run matlab script
    eng.quit()
    
    # h_head.close()
    Q.close()
    

#------------------------------------------------------------------------------
# Convert lakes .tif into a .shp, work with .shp results
#------------------------------------------------------------------------------
    if bool(lake_shapefiles):
        import rasterio
        import pandas as pd
        import geopandas as gpd
        from shapely.geometry import shape, JOIN_STYLE, MultiPolygon
        import math
        import itertools
        
        data= rxr.open_rasterio(f'{indir}lakes.tif',parse_coordinates=True,mode='r')
        r= [b for b in data.rio.bounds()]  # Read data bounds before reading data values
        region= [r[0],r[2],r[1],r[3]]
        # raster= rio.open_rasterio(f'{indir}lakes.tif',parse_coordinates=True,mode='r+')
        # raster= raster.squeeze().astype(np.float32).to_numpy()
        t= rasterio.transform.from_bounds(region[0],region[2],
                                      region[1],region[3],
                                      data.rio.width,
                                      data.rio.height)
        data=data.squeeze().astype(np.float32).to_numpy()
        data[~np.isnan(data)]=1
        data[np.isnan(data)]= 0
    
        poly, vals= [],[]
        for vec,value in rasterio.features.shapes(data,connectivity=8,transform=t):
            if value != 0:
                poly.append(shape(vec))
                vals.append(value)
        
        d= {'vals':vals,'geometry':poly}
        df= pd.DataFrame(d) # convert dataframe into a 2d numpy array
        geo_df= df.copy(deep=True)
        
        geo_df= gpd.GeoDataFrame(df,crs="EPSG:3031",geometry='geometry') #create geodataframe
        
        # iterate over pairs of polygons in the GeoDataFrame 
        buff= 500
        dist= 500
        
        col = ['geometry']
        res = gpd.GeoDataFrame(columns=col)
        
        for i, j in list(itertools.combinations(geo_df.index, 2)):
         distance= geo_df.geometry[i].distance(geo_df.geometry[j]) # distance between polygons i and j in the shapefile
         if distance < dist: 
             e = MultiPolygon([geo_df.geometry[i],geo_df.geometry[j]])
             fx = e.buffer(buff, 1, join_style=JOIN_STYLE.mitre).buffer(-buff, 1, join_style=JOIN_STYLE.mitre)
             res = res.append({'geometry':fx},ignore_index=True)
        
        # save the resulting shapefile   
        res.to_file(f'{indir}lakes.shp')
        


#-----------------------------------------------------------------------------
# Workflow for calculating hydraulic potential and simulating water routing
#----------------------------------------------------------------------------- 
def main(i):
    folder= f'S:/DronningMaudLand - Topographic uncertainty/_results_/DML_v3.4/{str(i).zfill(3)}/' 
    
    bed_path= folder+'SGSim.tif'
    # h_ice= folder+'h_ice.tif'  
    rema_path= 'S:/DronningMaudLand - Topographic uncertainty/Data/REMA/REMA_1km_customProj.tif'
    # bed_above_pmp= 'D:/Data/worked/basal_temperatures/bed_above_pmp.tif'
    
    outdir= f'S:/DronningMaudLand - Topographic uncertainty/_results_/hydro_03-24/{str(i).zfill(3)}/'
    workdir.makedir(outdir, delete_first=True)
    
    target_crs= pyproj.CRS(CRS.from_dict({'proj':"stere",'lat_ts': -71,'units': 'm',
                                      'lat_0':-90, 'lon_0':10, 'ellps': 'WGS84'}))
    
    #-----------------------------------------------------------------------------
    # Read in grids and manual check no extreme values/error
    #-----------------------------------------------------------------------------
    bed= rxr.open_rasterio(bed_path, parse_coordinates=True).squeeze()
    spatial_ref= bed.spatial_ref
    
    bed = xr.where(bed > -4000, bed, np.nan) 
    bed = xr.where(bed < 4000, bed, np.nan)
    bed['spatial_ref']= spatial_ref
    
    #-------------------------------------------------------
    isurf= rxr.open_rasterio(rema_path, parse_coordinates=True).squeeze()
    isurf= isurf.interp_like(bed)
    
    isurf = xr.where(isurf > -4000, isurf, np.nan) 
    isurf = xr.where(isurf < 4000, isurf, np.nan)
    isurf['spatial_ref']= spatial_ref
    
    #-------------------------------------------------------
    h_ice= isurf-bed # Calculate thickness
    
    h_ice= h_ice.interp_like(bed)
    
    h_ice = xr.where(h_ice > 0, h_ice, 0)   # set to 0 or np.nan
    h_ice = xr.where(h_ice < 4000, h_ice, 0)
    h_ice['spatial_ref']= spatial_ref
    
    
    # Save rasters-------------------------------------------
    if bool(saveRasters):
        bed.rio.to_raster(f'{outdir}bed.tif',driver='GTiff', nodata=np.nan)
        h_ice.rio.to_raster(f'{outdir}h_ice.tif',driver='GTiff', nodata=np.nan)
        isurf.rio.to_raster(f'{outdir}isurf.tif',driver='GTiff', nodata=np.nan)
    
    makeFigs={'Bed elevation':'bed','Surface elevation':'isurf','Ice thickness':'h_ice'}
    for name, rast in makeFigs.items():
        cfig.topo_figure(f'{outdir}/{rast}.tif',
                         f'{outdir}/{name}.png',
                         f'{name}',cmap='viridis',
                         hillshade=False, hs_ve= 0.05, contours=True)
  
    
    #--------------------------------------------------------------
    hydrology(f'{outdir}',bed,h_ice,isurf,spatial_ref=spatial_ref,tmp_mask=False)
    #--------------------------------------------------------------

    cfig.hydrology_figure(f'{folder}/bed.tif',  
                          f'{folder}{dir_id}/flow_accumulation.tif',
                          f'{folder}{dir_id}/lakes.tif',
                          f'{folder}{dir_id}/hydrology_figure.png',
                          'Bed elevation (m)', 
                min_FA=1500, min_lake_depth= 10,
                contours=True, plot_observed= False,
                plot_surveys=False, histogram=False,
                hillshade=True, hs_ve=0.05, cmap='gray',clip=False)
    
    cfig.topo_figure(f'{folder}{dir_id}/hyd_p.tif',
                     f'{folder}{dir_id}/hyd_p.png',
                     'Hydrostatic pressure (kPa)',cmap='viridis',
                     hillshade=False, hs_ve= 0.05, contours=True)
    
    cfig.topo_figure(f'{folder}{dir_id}/h_head.tif',
                     f'{folder}{dir_id}/h_head.png',
                     'Hydraulic head (m)',cmap='viridis',
                     hillshade=False, hs_ve= 0.05, contours=True)

    bed.close(); h_ice.close(); isurf.close();

    print (f'Completed hydrology workflow:.....{i}\i')
        



#--------------------------------------------------------------------------
if __name__ == "__main__":
    # Options and variables
    tmp_mask= False
    
    #--------------------------------------------------------------------------
    if bool(run_in_parallel):    # run in parallel, split into managable chunks
        from joblib import Parallel, delayed
        if imax-imin > in_parallel_max:
            runs= int((imax-imin)/in_parallel_max)
            step= int((imax-imin)/runs)
            sets= np.arange(imin,imax+1,step= step)
            for i,vals in enumerate(sets):
                if i<len(sets)-1:
                    start= sets[i]
                    stop= sets[i+1]
                    print(start,stop)
                    Parallel(n_jobs=step)(              # joblib parallel method
                         delayed(main)(i) for i in range(start,stop))
                    
            # [i for i in [n for n in sets]]
            # sets= np.linspace(imin,imax,retstep= in_parallel_max,endpoint=True)
            
        else:
            Parallel(n_jobs=imax-imin)(                     # joblib parallel method
                                  delayed(main)(i) for i in range(imin,imax)) 
    #--------------------------------------------------------------------------
    else:
        # for i in range(imin,imax):   # Loop through simulation folders 
            # main(i)
        [main(i) for i in range(imin,imax)]
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
    

