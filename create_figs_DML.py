#-----------------------------------------------------------------------------
"""
Functions for plotting various figures in Dronning Maud Land region of Antarctica

Created on Tue Sep 20 09:39:25 2022
@author: calvin.shackleton@npolar.no
"""
#-----------------------------------------------------------------------------
    
"""---------------------------------------------------------------------------
#----Topography figure--------------------------------------------------------
#--------------------------------------------------------------------------"""
def topo_figure(topo_path, out_path, cbar_label, setminmax=False,
                contours=False, plot_surveys=False,histogram=False,
                histpath= None,hillshade=False,hs_ve=0.05,
                cmap='gist_earth',clip=False,**kwargs):
    #-------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import rioxarray as riox
    import numpy as np
    import xarray as xr
    
    # Read .tif file, return bounds or set from input--------------------------
    raster= riox.open_rasterio(topo_path,parse_coordinates=True,mode='r')
    if bool(clip):
        region=clip.copy()
    else:
        r= [b for b in raster.rio.bounds()]  # Read data bounds before reading values
        region= [r[0],r[2],r[1],r[3]]
    data=raster.squeeze()
    raster.close(); raster= None;
          
    # Set up figure and plot--------------------------------------------------  
    fig, ax= figure_setup(region,**kwargs)
    
    # Hillshading ------------------------------------------------------------
    if bool(hillshade):
        from matplotlib.colors import LightSource

        ls = LightSource(azdeg=315, altdeg=45)
        # rgb = ls.shade(data.values, cmap=plt.cm.gist_earth, blend_mode='overlay',
        #                vert_exag=0.001)
        hs= ls.hillshade(data.values, vert_exag=hs_ve) # fraction=0.2
        hs= riox.raster_dataset.xarray.DataArray(data=hs, coords=data.coords)
        hillshade= hs.plot.pcolormesh(ax=ax, cmap='gray',  # viridis | coolwarm
                                add_labels=False,
                                add_colorbar=False)
    if bool(setminmax):
        vmin= setminmax[0]
        vmax= setminmax[1]
        # data= xr.where((data < vmin) | (data > vmax), np.nan, data)
        data= xr.where(data < vmin, np.nan, data)
        # data= xr.where(data > vmin, np.nan, data)
    else:
        vmin=data.min() #.round()
        vmax=data.max() #.round()
        # pass
        
    image= data.plot.pcolormesh(ax=ax, cmap=cmap,  # viridis | coolwarm
                                # vmin= data.min().round(), # specify min and max for colorscale
                                # vmax= data.max().round(),
                                vmin= vmin,
                                vmax= vmax,
                                add_labels=False,
                                add_colorbar=False,
                                alpha= 0.90)
        
    #----Add colorbar---------------------------------------------------------
    fig_size= fig.get_size_inches()[0] # return size since set outside function
    fig.set_figwidth(fig_size+(fig_size*0.2)) # account for new axes i.e. coloubar space
    # fig.set_figheight(fig_size+(fig_size*0.2))
    cax = fig.add_axes([ax.get_position().x1+0.012,
                        ax.get_position().y0,
                        0.04,  # colorbar width
                        ax.get_position().height])

    cbar= plt.colorbar(image,cax=cax) # Similar to fig.colorbar(im, cax = cax)
    cbar.ax.tick_params(labelsize=8)
    cax.set_title(cbar_label,fontsize=8)
    # cbar.set_label(cbar_label,fontsize=label_font_size)
   
    #----Plot radar surveys onto topo map------------------------------------- 
    if bool(plot_surveys):
        import pandas as pd
        surveys= pd.read_parquet('D:/Data/worked/data/h_ice_merged.parquet')
        ax.plot(surveys.x, surveys.y, ',k', markersize=1)
        
    #----Add in contours from REMA ice surface-------------------------------- 
    if bool(contours):
        ice_surface= riox.open_rasterio(
            r"S:\Data\Spatial_data\REMA\REMA_1km_dem.tif", 
            parse_coordinates=True).squeeze()

        cont= ice_surface.plot.contour(ax=ax,
                      levels=[500,1000,1500,2000,2500,3000],   
                     linewidths= 0.1,
                     alpha=0.6,
                     linestyles='solid',  # solid | dashed | dotted
                     colors='black',
                     add_labels=False,
                     )    # plot ice surface 
        
        # Add labels to contours
        ax.clabel(cont,colors=['black'],fontsize=3,
            manual=False,inline=True,  
            fmt=' {:.0f} '.format)
        
    # Histogram---------------------------------------------------------------
    if bool(histogram):
        grid_histogram(data,histpath,xylim=['min','max','min','max'],
                   density=False, xlabel='value', ylabel='count',lfont_size=8,
                   bins='auto',histtype='stepfilled',color='lightsteelblue')
       
    plt.savefig(out_path,
                transparent=False,
                dpi= 600,
                bbox_inches="tight",
                # dpi=height/fig.get_size_inches()[1]
                )
    # return fig, ax
    plt.close()
    data.close(); data= None;
    
    
"""---------------------------------------------------------------------------
# Plot point data-------------------------------------------------------------
#--------------------------------------------------------------------------"""
def plot_data_points(data_path, out_path, region,
                     dmap={'x':'x','y':'y','z':'z'},
                     title='',cbar_label='Values',clip=False,background=False,**kwargs):
    
    import matplotlib.pyplot as plt
    # import matplotlib.colors as mcolors
    import pandas as pd
    # import numpy as np
    
    if isinstance(data_path,str):
        if data_path.endswith('.parquet'):
            data= pd.read_parquet(data_path)   # Load point data
            data= data.reset_index(drop=True)
        elif data_path.endswith('.csv'):
            data= pd.read_csv(data_path, low_memory=False)   # Load point data
    elif isinstance(data_path,pd.DataFrame):
        data= data_path.copy()   # check if input is DataFrame, copy data
    
    data= data.rename(dmap,axis=1)
    data= data[['x','y','z']]
    fig, ax = figure_setup(region,**kwargs)
    
    # Results should be in standardized x,y,z dataframe format
    points = plt.scatter(data.x, data.y, c=data.z, cmap='viridis',
                     vmin= data.z.min(), vmax= data.z.max(),
                     marker=".", s= 0.6, edgecolors='none')     # scatter plot for location map
    plt.title(title)  # add plot title
    
    #----Add colorbar---------------------------------------------------------
    fig_size= fig.get_size_inches()[0] # return size since set outside function
    fig.set_figwidth(fig_size+(fig_size*0.2)) # account for new axes i.e. colourbar space
    
    cax= fig.add_axes(rect=(ax.get_position().x1+0.012,
                            ax.get_position().y0,
                            0.04,
                            ax.get_position().height))
    cbar= plt.colorbar(points,cax=cax,fraction=0.1,ax=ax) # Similar to fig.colorbar(im, cax = cax)
    cbar.ax.tick_params(labelsize=8)
    cax.set_title(cbar_label,fontsize=8)
    # cbar.set_label(cbar_label,fontsize=8,rotation=270,-1) # colour bar labels
    
    if bool(clip):
        from cartopy.crs import SouthPolarStereo
        ax.set_extent(clip,crs=SouthPolarStereo(true_scale_latitude='-71'))
    
    # plt.axis('scaled')
    plt.savefig(out_path,
                transparent=True,
                bbox_inches="tight",
                dpi= 600)
    plt.close()
    
"""---------------------------------------------------------------------------
#----Hydrology figure--------------------------------------------------------
#--------------------------------------------------------------------------"""
def hydrology_figure(Q_path, S_path, L_path, out_path, cbar_label, 
                min_FA=2000, min_lake_depth=5, contours=False, plot_observed= False,
                plot_surveys=False,histogram=False,hillshade=False,
                hs_ve=0.5,cmap='gray',clip=False):
    #-------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import rioxarray as riox
    import numpy as np
    import cmocean.cm as cm
    
    # Read .tif file, return bounds ------------------------------------------
    data= riox.open_rasterio(Q_path,parse_coordinates=True,mode='r')
    r= [b for b in data.rio.bounds()]  # Read data bounds before reading data values
    region= [r[0],r[2],r[1],r[3]]
    data=data.squeeze()
          
    # Set up figure and plot--------------------------------------------------  
    fig, ax= figure_setup(region)
    
    # Hillshading ------------------------------------------------------------
    if bool(hillshade):
        from matplotlib.colors import LightSource

        ls = LightSource(azdeg=315, altdeg=45)
        # rgb = ls.shade(data.values, cmap=plt.cm.gist_earth, blend_mode='overlay',
        #                vert_exag=0.001)
        hs= ls.hillshade(data.values, vert_exag=hs_ve) # fraction=0.2
        hs= riox.raster_dataset.xarray.DataArray(data=hs, coords=data.coords)
        hillshade= hs.plot.pcolormesh(ax=ax, cmap='gray',  # viridis | coolwarm
                                add_labels=False,
                                add_colorbar=False)

    topo= data.plot.pcolormesh(ax=ax, cmap=cmap,  # viridis | coolwarm
                                vmin= data.min().round(),  # specify min and max for colorscale
                                vmax= data.max().round(),
                                add_labels=False,
                                add_colorbar=False,
                                alpha= 0.95)
    
    # Read in lakes and streams + plot----------------------------------------
    if S_path.endswith('.tif'):
        streams= riox.open_rasterio(S_path,parse_coordinates=True,mode='r').squeeze()
        streams= streams.astype(float) # sqrt for nice map shading
        streams.values[streams.values<min_FA]=np.nan
        streams.values= np.sqrt(streams.values)
    
        s_plot= streams.plot.pcolormesh(ax=ax, cmap=cm.ice_r,  # viridis | coolwarm
                                    vmin= streams.min().round(),  # specify min and max for colorscale
                                    vmax= streams.max().round(),
                                    # norm='log',
                                    add_labels=False,
                                    add_colorbar=False,
                                    # alpha= 0.70,
                                    )
    elif S_path.endswith('.shp'):
        import geopandas as gpd
        # from rasterio.plot import plotting_extent
        
        stream_v=gpd.read_file(S_path).set_crs(epsg=3031)
        # plot_ext= plotting_extent(data, transform= data.rio.transform())
        stream_v.plot(ax=ax, linewidth= stream_v.streamorder*0.2,
                      color='#2460A7FF') #  2460A7FF | 8BBEE8F
        
    lakes= riox.open_rasterio(L_path,parse_coordinates=True,mode='r').squeeze()
    lakes= lakes.astype(float)
    lakes.values[lakes.values<min_lake_depth]=np.nan
    
    # plt.imshow(lakes, cmap= 'Blues')
    l_plot= lakes.plot.pcolormesh(ax=ax, cmap=cm.ice_r,  # viridis | coolwarm
                                # vmin= data.min().round(),  # specify min and max for colorscale
                                # vmax= data.max().round(),
                                add_labels=False,
                                add_colorbar=False,
                                # alpha= 0.90,
                                # zorder=3,
                                )
   
    #----Add colorbar---------------------------------------------------------
    fig_size= fig.get_size_inches()[0] # return size since set outside function
    fig.set_figwidth(fig_size+(fig_size*0.2)) # account for new axes i.e. coloubar space
    # fig.set_figheight(fig_size+(fig_size*0.2))
    cax = fig.add_axes([ax.get_position().x1+0.012,
                        ax.get_position().y0,
                        0.04,  # colorbar width
                        ax.get_position().height])

    cbar= plt.colorbar(topo,cax=cax) # topo | l_plot | s_plot
    cbar.ax.tick_params(labelsize=8)
    cax.set_title(cbar_label,fontsize=8)
    # cbar.set_label(cbar_label,fontsize=label_font_size) 
    
    #----Plot radar surveys onto topo map------------------------------------- 
    if bool(plot_surveys):
        import pandas as pd
        surveys= pd.read_parquet('D:/Data/worked/data/h_ice_merged.parquet')
        ax.plot(surveys.x, surveys.y, ',k', markersize=1)
        
    #----Plot observed lakes onto topo map------------------------------------- 
    if bool(plot_observed):
        import pandas as pd
        import pyproj
        lakes_siegert = pd.read_csv('D:\Data\misc\sgl_siegert_2005.txt');
        lakes_popov = pd.read_csv('D:\Data\misc\lakes_popov_masolov_2007.txt');
        lakes_karlsson= pd.read_csv('D:\Data\misc\lakes_karlsson.txt');
        
        projection = pyproj.Proj("epsg:3031") # Antarctic Polar Stereographic
        
        x,y = projection(lakes_siegert.lon,lakes_siegert.lat*-1)  # Project lat lon
        ax.scatter(x,y,marker='*',c='r',s=0.1)
        x,y = projection(lakes_popov.lon,lakes_popov.lat)
        ax.scatter(x,y,marker='*',c='r',s=0.1)
        x,y = projection(lakes_karlsson.lon,lakes_karlsson.lat)
        ax.scatter(x,y,marker='*',c='r',s=0.1)
        
    #----Add in contours from REMA ice surface-------------------------------- 
    if bool(contours):
        ice_surface= riox.open_rasterio(
            r"S:\Data\Spatial_data\REMA\REMA_1km_dem.tif", 
            parse_coordinates=True).squeeze()

        cont= ice_surface.plot.contour(ax=ax,
                     levels=[500,1000,1500,2000,2500,3000],  
                     linewidths= 0.1,
                     alpha=0.6,
                     linestyles='solid',   # solid | dashed | dotted
                     colors='black',
                     add_labels=False,
                     )    # plot ice surface 
        
        # Add labels to contours
        ax.clabel(cont,colors=['black'],fontsize=4,
            manual=False,inline=True,  
            fmt=' {:.0f} '.format)
        
    # Histogram---------------------------------------------------------------
    if bool(histogram):
        hist_path= out_path.split('.')
        hist_path.insert(1,'_hist.')
        hist_path=''.join(hist_path)
        grid_histogram(data,hist_path,xylim=['min','max','min','max'],
                   density=False, xlabel='value', ylabel='count',lfont_size=8,
                   bins='auto',histtype='stepfilled',color='lightsteelblue')
        
    if bool(clip):
        from cartopy.crs import SouthPolarStereo
        ax.set_extent(clip,crs=SouthPolarStereo(true_scale_latitude='-71'))
        
    plt.savefig(out_path,
                transparent=False,
                dpi= 600,
                bbox_inches="tight",
                # dpi=height/fig.get_size_inches()[1]
                )
    plt.close()
    
"""---------------------------------------------------------------------------
--Utility functions-----------------------------------------------------------
--------------------------------------------------------------------------"""
def figure_setup(region,lfont_size=8, fig_size=8, fig_dpi=600,
                 background=False,stations=False,ice_drain=False,ground_line=False, **kwargs):
#-----------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    from cartopy.crs import SouthPolarStereo
    # import cartopy
    from numpy import linspace, arange
    # from rasterio.crs import CRS
    import pyproj
    # import proj2cartopy
    plt.rcdefaults()
    
    crs = SouthPolarStereo(true_scale_latitude='-71',central_longitude=10.0)
    # crs= pyproj.CRS(CRS.from_dict({'proj':"stere",'lat_ts': -71,'units': 'm',
    #                                   'lat_0':-90, 'lon_0':10, 'ellps': 'WGS84'}))
    fig = plt.figure(figsize=(fig_size, fig_size))
    # fig.set_visible(False)
    ax = plt.axes(projection=crs)
    ax.set_extent(region, crs=crs)
    
    #----Add background raster------------------------------------------------
    if bool(background):
        import rioxarray as rxr
        grid= rxr.open_rasterio(kwargs['background']).squeeze()
        grid= grid.rio.reproject(crs).rio.clip_box(region[0],region[2],region[1],region[3])
        gridplot= grid.plot.imshow(ax=ax,cmap='gist_heat_r',vmin=0,vmax=800,alpha=0.4,
                         add_colorbar =False,add_labels=False) #**{'font_size':20}
        
        # fig, ax= add_colorbar(fig,ax,gridplot,"Flow (m/yr)")
    
    #----Adjust labels and plot title*------------------------------------
    ax.set_ylabel('Northing (km)',fontsize=lfont_size)
    ax.set_xlabel('Easting (km)',fontsize=lfont_size)
    ax.set_title(" ")
    #---------------------------------------------------------------------
    
    # xticks= linspace(region[0],region[1],6) # min, max, n-steps
    # yticks= linspace(region[2],region[3],6)  
    xticks= arange(region[0],region[1],200000)
    yticks= arange(region[2],region[3],100000)
    ax.set_xticks(xticks, crs=crs)
    ax.set_yticks(yticks, crs=crs)
    ax.set_xticklabels((xticks/1000).astype(int))
    ax.set_yticklabels((yticks/1000).astype(int))
    plt.xticks(fontsize=lfont_size)
    plt.yticks(fontsize=lfont_size)
    # ax.yaxis.set_label_position("right")  # Axes labels e.g. top, right
    # ax.yaxis.tick_right()
    # ax.xaxis.set_label_position("top")
    # ax.xaxis.tick_top()
    ax.tick_params(axis='both',direction='out',
                left=True, bottom=True, right=True, top=True)
    

    #----Add shapefiles to plot-----------------------------------------------
    # Antarctic ice drainage boundries
    if bool(ice_drain):
        import geopandas as gpd
        ice_file='S:/QGIS/Quantarctica3/Glaciology/MEaSUREs Antarctic Boundaries/IceBoundaries_Antarctica_v2.shp'
        gdf_ice= gpd.read_file(ice_file).set_crs(epsg=3031)
        gdf_ice= gdf_ice.to_crs(crs)
        feature= 'Subregions' # NAME|Asso_shelf|Regions|Subregions
        gdf_ice.plot(column=feature,ax=ax,edgecolor='k',facecolor='none',
                     linestyle=(0, (5, 5)),linewidth=0.6,alpha=0.99) #f0bebd
    
    # Grounded ice margin (RINGS)
    if bool(ground_line):
        import geopandas as gpd
        GL_file='S:/QGIS/Quantarctica3/QA-RINGS/GroundingLine/RINGS_workshop_grounded_2022.shp'
        gdf_GL= gpd.read_file(GL_file).set_crs(epsg=3031)
        gdf_GL= gdf_GL.to_crs(crs)
        feature= 'Area_km2' # Id|Area_km2
        gdf_GL.plot(column=feature,ax=ax,edgecolor='k',facecolor='none',linewidth=0.6,alpha=0.99)
     
    # Plot station locations -------------------------------------------------
    if bool(stations):
        import pyproj
        import pandas as pd

        projection= pyproj.Proj(crs)
        plot_df= pd.read_csv(r'S:\DronningMaudLand - Topographic uncertainty\other\stations.csv') 
        plot_df['x'], plot_df['y'] = projection(plot_df.Longitude, plot_df.Latitude)   # Project coordinates
        # proj_coords = wgs84(*plot_coords)         # Project coordinates
        # ax.plot(*proj_coords, 'dr',markersize=1)  # markersize=2
        ax.plot(plot_df.x.values,plot_df.y.values, 'dr',markersize=1.8)
        
        for i, label in enumerate(plot_df.Station):
            ax.annotate(label,(plot_df.x[i]+10000,plot_df.y[i]+10000),color='#666666',
                    fontsize=lfont_size)  # backgroundcolor='w'
    return fig, ax

#-----------------------------------------------------------------------------
# Create histogram, option to add to plotted grid-----------------------------
#-----------------------------------------------------------------------------
def grid_histogram(grid,hist_path, xylim=['min','max','min','max'],logyaxis=False,
                   density=False, xlabel='value', ylabel='count',lfont_size=8,
                   bins='auto',histtype='stepfilled',color='lightsteelblue',
                   xgrid_interval=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    plt.rcdefaults()
    
    d=  np.hstack(grid)
    df= pd.DataFrame(d, columns=['data'])
    df= df.dropna(how='all')
    df.reset_index()
    
    h_fig= plt.figure(figsize=(3,3))
    h_ax= plt.axes()
    plt.hist(df, density=density,   # density vs frequency
             bins=bins,      # auto | sqrt | fd
             histtype= histtype,   # bar | step | stepfilled
             # label=[],        # string or sequence of strings
             color=color)
    std= float(df.std())  # Return standard deviation
    
    plt.xlabel(xlabel); plt.ylabel(ylabel);
    # plt.title('Histogram of IQ')
    if bool(logyaxis):
        plt.yscale("log")               # plot y axis on a log scale
    xylim= axes_limits(h_ax, xylim)     # Convert limits based on specified
    # plt.ylim(ymin=xylim.ymin)                # change x limits/range
    plt.xlim(xmin=xylim.xmin[0], xmax=xylim.xmax[0]) # change x limits/range  [-0.3,0.3]
    # plt.xlim(xmin=std*-1, xmax=std)    # change x limits/range
    plt.xticks(fontsize=lfont_size)
    plt.yticks(fontsize=lfont_size)

    # Add the grid
    if bool(xgrid_interval):
        loc = plticker.MultipleLocator(base=xgrid_interval)
        h_ax.xaxis.set_minor_locator(loc)
    # h_ax.yaxis.set_major_locator(loc)
    h_ax.grid(which='both', axis='both', linestyle='--',
              linewidth=0.4, alpha=0.5)
    
    # Save figure
    h_fig.savefig(hist_path,
                format='png',
                transparent=True,
                bbox_inches="tight",
                dpi= 300)
    plt.close()
    
#-----------------------------------------------------------------------------
# Axes limits function: returns appropriate axes limits from user input-------
#-----------------------------------------------------------------------------
def axes_limits(axes_h, xylim=['min','max','min','max']):
    import pandas as pd
    ax_lims= pd.DataFrame(axes_h.get_xlim()+axes_h.get_ylim()).T
    ax_lims.columns=['xmin','xmax','ymin','ymax']
    
    for i,lim in enumerate(xylim):
        col= ax_lims.columns[i]
        if type(lim) == str:
            continue
        else:
            ax_lims[col]=lim
    return ax_lims

def add_colorbar(fig,ax,data,title):
    import matplotlib.pyplot as plt
    fig_size= fig.get_size_inches()[0] # return size since set outside function
    fig.set_figwidth(fig_size+(fig_size*0.2)) # account for new axes i.e. colourbar space
    
    cax= fig.add_axes(rect=(ax.get_position().x1+0.012,
                            ax.get_position().y0,
                            0.04,
                            ax.get_position().height))
    cbar= plt.colorbar(data,cax=cax,fraction=0.1,ax=ax) # Similar to fig.colorbar(im, cax = cax)
    cbar.ax.tick_params(labelsize=8)
    cax.set_title(title,fontsize=8)
    
    return fig,ax
   
        
        