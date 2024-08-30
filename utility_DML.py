#------------------------------------------------------------------------------
"""
Utility script for kriging + sequential gaussian simulation
    - Import data, write geotiff, grid point data, local outlier removal,
      reverse normal score, calculate bed elevation, plot figures

    Created on Fri Jul  8 12:44:11 2022
    @author: calvin.shackleton@npolar.no

"""
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geostatspy.geostats as geostats 
import create_figs_DML as cfig

#-----------------------------------------------------------------------------
# Data: read file and clip ---------------------------------------------------
#-----------------------------------------------------------------------------
def read_clip_data(inpath, folder, region=[], clip=False,
                   dmapin={'x':'x','y':'y','z':'z'},
                   dmapout={'x':'x','y':'y','z':'z'}):
    if isinstance(inpath,str):
        if inpath.endswith('.parquet'):
            data= pd.read_parquet(inpath)   # read point data
            # data= data.reset_index(drop=True)
        elif inpath.endswith('.csv'):
            data= pd.read_csv(inpath, low_memory=False)   # read point data
    elif isinstance(inpath,pd.DataFrame):
        data= inpath.copy(deep=True)   # check if input is DataFrame, copy data
    
    data= data.rename(dmapin,axis=1)
    
    if bool(clip):
        data= data.loc[region[0]<=data.x]  # Exclude data beyond region of interest
        data= data.loc[data.x<=region[1]]  # ...
        data= data.loc[region[2]<=data.y]  # ...
        data= data.loc[data.y<=region[3]]  # ...
    
    data= data.rename(dmapout,axis=1)
    # data= data[['x','y','z']]
    
    data.to_parquet(folder+'df_all.parquet')
    cfig.plot_data_points(folder+'df_all.parquet',
                          f'{folder}/All_points_map.png',
                          region, dmap={'X':'x','Y':'y','Bed':'z'},
                          title='All data points',
                          cbar_label='Elevation (m)')
    
    return data

#-----------------------------------------------------------------------------
# Subset data, removing data within input region (like clipping out data in box)
#-----------------------------------------------------------------------------
def region_subset(df,region,
                  dmapin={'x':'x','y':'y','z':'z'},
                  dmapout={'x':'x','y':'y','z':'z'}):
    ''' DataFrame must have x and y columns
        Region format x1,x2,y1,y2 '''
        
    df= df.rename(dmapin,axis=1)
    df= df.loc[(region[0]<=df.x)&
               (df.x<=region[1])&
               (region[2]<=df.y)&
               (df.y<=region[3])] 
    df= df.rename(dmapout,axis=1)
    return df

#-----------------------------------------------------------------------------
# Data: normal score transformation + downsample -----------------------------
#-----------------------------------------------------------------------------
def sample_Ntransform_data(df_in,region,folder,method='random',
                           fraction_or_spacing=100,res=500,
                           center_coords=False,drop_coords=True,replace=False):
        
# Downsample and grid using verde block reduction-----------------------------
    if method=='decimate':
        import verde as vd
        reducer= vd.BlockReduce(reduction=np.median, spacing=fraction_or_spacing,
                              region=region, center_coordinates=center_coords,
                              drop_coords=drop_coords)
        coordinates, thickness = reducer.filter(coordinates=((df_in.X,df_in.Y)),
                                                data=df_in.Bed)
        df_samp= pd.DataFrame({'X':coordinates[0],
                               'Y':coordinates[1],
                               'Bed':thickness})
        df_validate= df_in.copy() # if decimating data, all data used as validate

# Randomly downsample then grid data------------------------------------------
    if method=='random':
        df_samp = df_in.sample(frac=fraction_or_spacing,
                               replace=replace, random_state=1)
        df_validate= df_in.drop(df_samp.index) # Drop sampled data, save remainder for validation
    
        import GlacierStats as gs
        df_samp, grid_matrix, rows, cols = gs.grid_data(
            df_samp, 'X', 'Y', 'Bed', res=res) # grid data
        
        df_samp = df_samp[df_samp["Z"].isnull() == False]  # remove coordinates with NaNs
        df_samp = df_samp.rename(columns = {"Z": "Bed"}) # rename column for consistency
        
# Use entire points dataset and grid------------------------------------------
    if method=='none':
        import GlacierStats as gs
        df_samp, grid_matrix, rows, cols = gs.grid_data(
            df_in, 'X', 'Y', 'Bed', res=res)               # grid data 
        df_samp = df_samp[df_samp["Z"].isnull() == False]  # remove coordinates with NaNs
        df_samp = df_samp.rename(columns = {"Z": "Bed"})   # rename column for consistency
        
        df_validate= df_in.copy() # if gridding full data, all data used as validate
        
# Normal score transformation + plot results/stats----------------------------
    df_samp['Nbed'], tvbed, tnsbed = geostats.nscore(df_samp,'Bed') 
    
# Plot data and transformed data histograms-----------------------------------
    fig= plt.figure(figsize=(15,10))
    gs= fig.add_gridspec(1,2)
    
    # plot original bed histogram
    ax1=fig.add_subplot(gs[0,0])
    plt.hist(df_samp['Bed'], facecolor='red',
             bins=50,alpha=0.3,edgecolor='none',label='data') 
    plt.xlabel('Elevation (m)'); plt.ylabel('Count')
    plt.title('Sampled Elevation (m)')
    text= f"min: {df_samp['Bed'].min()}\nmax: {df_samp['Bed'].max()}"
    plt.text(0.5, 0.8, text,transform=ax1.transAxes)
    plt.legend(loc='upper left')
    plt.grid(True,linewidth=0.4,linestyle='--',alpha=0.6)
    
    # plot normal score bed histogram (with weights)
    ax1=fig.add_subplot(gs[0,1])
    plt.hist(df_samp['Nbed'], facecolor='red',bins=50,
             alpha=0.3,edgecolor='none')
    plt.xlabel('Nscore Elevation'); plt.ylabel('')
    plt.title('Nscore: Elevation')
    plt.grid(True,linewidth=0.4,linestyle='--',alpha=0.6)
    
    plt.gcf().savefig(f'{folder}/histogram_data_nscore.png',dpi= 300)
    plt.close()
            
# Save output to file (convert ndarray to dataframe, save as parquet)---------
    df_samp.to_parquet(folder+'df_samp.parquet')  
    df_validate.to_parquet(folder+'df_validate.parquet')
    pd.DataFrame(tvbed,columns=['tvbed']).to_parquet(folder+'tvbed.parquet')
    pd.DataFrame(tnsbed,columns=['tnsbed']).to_parquet(folder+'tnsbed.parquet')
    
    # Plot data points on a map-----------------------------------------------
    for dplot in ['df_samp','df_validate']:  # plot all versions
        cfig.plot_data_points(folder+f'{dplot}.parquet',
                              f'{folder}/{dplot}_map.png',
                              region, dmap={'X':'x','Y':'y','Bed':'z'},
                              title=f'{dplot} data points [Spacing:{fraction_or_spacing}]',
                              cbar_label='Elevation (m)')
    
    fprint('Sampled + normal transformed data')
    return df_samp, tvbed, tnsbed

#-----------------------------------------------------------------------------
# Make XY coordinate grid from corner points----------------------------------
#-----------------------------------------------------------------------------
def pred_grid(xmin, xmax, ymin, ymax, pix):
    cols = np.rint((xmax - xmin)/pix); rows = np.rint((ymax - ymin)/pix)  # number of rows and columns
    x = np.arange(xmin,xmax,pix); y = np.arange(ymin,ymax,pix) # make arrays

    xx, yy = np.meshgrid(x,y) # make grid
    yy = np.flip(yy) # flip upside down

    # shape into array
    x = np.reshape(xx, (int(rows)*int(cols), 1))
    y = np.reshape(yy, (int(rows)*int(cols), 1))

    Pred_grid_xy = np.concatenate((x,y), axis = 1) # combine coordinates
    return Pred_grid_xy

#-----------------------------------------------------------------------------
# Grid input xyz data---------------------------------------------------------
#-----------------------------------------------------------------------------
def points_to_grid(x,y,z,region,res):
    """Requires regularly spaced points e.g. kriging or SGSim point output
        pass only data series to function, not full dataframe"""

    xmin = region[0]; xmax = region[1]     # range of x values
    ymin = region[2]; ymax = region[3]     # range of y values
    ylen = int((ymax - ymin)/res)
    xlen = int((xmax - xmin)/res)
    grid= np.reshape(z.to_numpy(), (ylen, xlen))
    xx= np.linspace(start= xmin, stop= xmax, # divide x dimension
            endpoint=True, num=xlen)
    yy= np.linspace(start=ymin, stop= ymax, # divide y dimension
            endpoint=True, num=ylen)
    
    # xx = np.reshape(x_coords.to_numpy(), (int(ylen), int(xlen)))
    # yy= np.reshape(y_coords.to_numpy(), (int(ylen), int(xlen)))
    
    fprint('Points to grid')
    return grid, xx, yy

#-----------------------------------------------------------------------------
# Write out GeoTiff from xyz grid---------------------------------------------
#-----------------------------------------------------------------------------
def write_geotiff(outfile,xx,yy,grid,region):
    from rasterio.crs import CRS
    import rasterio
    import xarray as xr
    import rioxarray 
    import pyproj
    
    grid_array= xr.DataArray(data=grid,
                             dims=['y','x'],
                             coords=(yy[:], xx[:]))
    
    # crs= CRS.from_epsg(3031)
    crs= pyproj.CRS(CRS.from_dict({'proj':"stere",'lat_ts': -71,'units': 'm',
                                      'lat_0':-90, 'lon_0':10, 'ellps': 'WGS84'}))

    
    # Define transform 
    t= rasterio.transform.from_bounds(region[0],region[2],
                                      region[1],region[3],
                                      grid_array.rio.width,
                                      grid_array.rio.height)
    # t= rasterio.transform.from_bounds(xul,ylr,xlr,yul,
    #                                   grid_array.rio.width,
    #                                   grid_array.rio.height)
    # t= rasterio.transform.from_origin(xul,yul,500,500)
    # t= Affine(transform[0],transform[1],transform[2],  # modify to adjust to negative
    #                         transform[3],transform[4],transform[5])
    
    grid_array= grid_array.rio.write_crs(crs,inplace=True) 
    grid_array= grid_array.rio.write_grid_mapping('spatial_ref',inplace=True)
    
    # grid_array.to_netcdf(folder+'grid.nc')      # write .netcdf
    raster = rasterio.open(outfile,   # write .tiff
                            'w', driver='GTiff',
                            height = grid_array.shape[0],
                            width = grid_array.shape[1],
                            count=1, dtype=str(grid_array.dtype),
                            crs= crs,
                            # indexes=1,
                            transform=t)  
    raster.write(grid_array.astype(rasterio.float64),1)
    raster.close(); raster = None;
    fprint('Write out GeoTiff')
    

#-----------------------------------------------------------------------------
# Reverse normal score transformation-----------------------------------------
#-----------------------------------------------------------------------------
def reverse_norm_score(data,z,tvbed,tnsbed):
# create dataframes for back transform function
    df_z= pd.DataFrame(data, columns=[z])
    
    # Transformation parameters (first convert to numpy if not already)
    if type(tvbed).__module__ != np.__name__:
        tvbed= tvbed.to_numpy()
        tnsbed= tnsbed.to_numpy()
    
    vr= tvbed;  # Transformation table original values
    vrg= tnsbed # Transformation table transformed variable
    ltail= 1;  utail= 1; # Lower/Upper tail values
    zmin= -4;  zmax= 4  # Trimming limits
    ltpar= -1000; utpar= 1000 # Lower/Upper extrapolation parameter
    
    trans_z = geostats.backtr(df_z, z, vr, vrg, zmin, zmax,
                              ltail, ltpar, utail, utpar)
    fprint('Reverse normal score')
    return trans_z 

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def fprint(process='Process'):
    from datetime import datetime
    time= datetime.now().strftime("%H:%M:%S")
    print ('{:.<35}{}'.format(f'{process}',f'{time} (h:m:s)'))

#------------------------------------------------------------------------------

