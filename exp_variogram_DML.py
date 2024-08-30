#-----------------------------------------------------------------------------
"""
Compute experimental variograms and model variograms for clustered data
    - K-means clusters for data based on values and x,y
    - compute experimental variogram for each cluster
    - model variogram for each cluster

12/06/2023
@author: calvin.shackleton@npolar.no
"""
#-----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skgstat as skg
from skgstat import models
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer 
import pyproj
import create_figs_DML as cfig
from utility_DML import region_subset
from rasterio.crs import CRS
# from scipy.optimize import curve_fit
from utility_DML import fprint

#-----------------------------------------------------------------------------
# Run functions: True | False 
import_data_norm_resample_cluster= True  
calc_experimental_variogram= True
anisotropic_variogram = False
fit_model_to_variogram= True

areatag= 'DML'  # east|west|DML
#--------------------------------------

target_crs = pyproj.CRS(CRS.from_dict({'proj':"stere",'lat_ts': -71,'units': 'm',
                                      'lat_0':-90, 'lon_0':10, 'ellps': 'WGS84'}))
if areatag =='west':
    region= [-510000,240000,-480000,130000] # Custom CRS [west]
elif areatag =='east':
    region= [230000,1190000,-480000,160000] # Custom CRS [east]
elif areatag == 'DML':
    region= [-700000,1300000,1800000,2230000] # Custom CRS full region
    
# Options and variables --------------------------------------
drive= 'D'
foldertag= f'{areatag}_v3.0'
sample_method= 'Verde median reduction filter' 
fraction_or_spacing= 100  # 0.0-1.0 |or| 500
downsample_fraction= 0.5
clip_data_with_shapefile= True
shapefile_path= f'{drive}:\QGIS\Quantarctica3\QA-RINGS\GroundingLine\RINGS_workshop_grounded_2022.shp'
center_coords= False 
n_clusters= 30

Azimuth= 0
lag_dist = 1000
maxlag = 80000 # 12000
n_lags= int(maxlag/lag_dist)
    
#-----------------------------------------------------------------------------
indir= f'{drive}:/Data/DronningMaudLand/'
outdir= f'{drive}:/Data/DronningMaudLand/variogram/{foldertag}/'
if os.path.exists(outdir):
    # shutil.rmtree(outdir)
    # os.makedirs(outdir)
    pass
else:
    os.makedirs(outdir)
fprint('Start'); print(outdir);

exp_v_dir= f'{outdir}exp_variograms/'

# inCSV= f'{indir}bed_data_full.csv'
survey_data= f'{indir}data/bed_data_{areatag}.parquet'

#-----------------------------------------------
def clip_data_to_polygon(indata,shapefile_path):
    import geopandas as gpd
    geodf= gpd.GeoDataFrame(data= indata,
                            crs= target_crs,
                            geometry= gpd.points_from_xy(
                                indata.X, indata.Y)) # to geopandas dataframe
    
    GL_shp= gpd.read_file(shapefile_path)
    # GL_mask=gpd.GeoDataFrame.explode(GL_shp,index_parts=True) # shapefile > geodataframe
    GL_mask= GL_shp.to_crs(target_crs) # reproject input geometry to custom crs
    # GL_mask= GL_mask.buffer(1000)
    outdf= gpd.clip(geodf, GL_mask) # Mask out non-terrestrial data
    outdf= outdf.drop(columns=['geometry'])
    
    fprint('Clipped data using shapefile')
    
    return outdf
    
#-----------------------------------------------------------------------------
# Import data, normal score transformation + downsample ----------------------
#-----------------------------------------------------------------------------
def data_preprocessing():
    
    # data= util.read_clip_data(inCSV, outdir, region,
    #                          dmapin={'x':'x','y':'y','bed':'z'},
    #                          dmapout={'x':'X','y':'Y','z':'Bed'})
    
    # df_samp, tvbed, tnsbed= util.sample_Ntransform_data(
    #     data,region,outdir,method= sample_method, drop_coords=False,
    #     fraction_or_spacing= fraction_or_spacing, center_coords=center_coords)
    
    # Read data + clip to ice shelves/coastline/groudning line------------
    df_bed= pd.read_parquet(survey_data,columns=['x','y','bed'])
    
    df_bed= region_subset(df_bed, region,
                          dmapin={'x':'x','y':'y','bed':'z'},
                          dmapout={'x':'X','y':'Y','z':'Bed'})
    
    if bool(clip_data_with_shapefile):
        df_bed= clip_data_to_polygon(df_bed,shapefile_path)
    
    cfig.plot_data_points(df_bed,f'{outdir}bed_data_all.png',
                          [df_bed.X.min(),df_bed.X.max(),
                           df_bed.Y.min(),df_bed.Y.max()],
                      dmap={'X':'x','Y':'y','Bed':'z'},title='',
                      cbar_label='Bed Elevation (m)',**{'ground_line':True,'stations':True})
    
    # -----------------Verde data decimate function----------------------------
    import verde as vd
    reducer= vd.BlockReduce(reduction=np.median, spacing=fraction_or_spacing,
                              region=region, center_coordinates=False,
                              drop_coords=False)
    coordinates, data = reducer.filter(coordinates=((df_bed.X,df_bed.Y)),
                                                data= df_bed.Bed)
    df_samp= pd.DataFrame({'X':coordinates[0],
                           'Y':coordinates[1],
                           'Bed':data})
    cfig.plot_data_points(df_samp,f'{outdir}bed_data_sampled.png',region,
                      dmap={'X':'x','Y':'y','Bed':'z'},
                      cbar_label='Bed Elevation (m)',
                      **{'ground_line':True,'stations':True})
    
    df_samp= df_samp.dropna()  # K-means cannot handle NaN
    fprint('Decimated ice thickness data')
    
    
    kmeans = KMeans(n_clusters= n_clusters,init='k-means++',random_state=0,
                    ).fit(df_samp[['X','Y','Bed']])
    df_samp['K'] = kmeans.labels_  # make column in dataframe with cluster name
    
    # df_samp.to_parquet(outdir+'df_samp.parquet')
    fprint('K-means clustering assigned')
    
    #-----------------------------------------------------------------------------
    # normal score transformation
    data = df_samp['Bed'].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
    df_samp['Nbed'] = nst_trans.transform(data)
    
    df_samp.to_parquet(outdir+'df_samp.parquet')
    df_samp.to_csv(outdir+'df_samp.csv',index=False)
    
    cfig.plot_data_points(df_samp,
                  f'{outdir}k-means_gridded.png',region,
                  dmap={'X':'x','Y':'y','K':'z'},
                  title='',cbar_label='Cluster')
    fprint('Normal-score transformation')

#-----------------------------------------------------------------------------
# Compare model fits for experimental variogram and return best parameters
#-----------------------------------------------------------------------------
def find_best_model(coords, values, n_lags, maxlag, outdir, weighting='linear',plot=True):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # Determine best model type (you can add more models to compare)
    best_model_name = None
    best_weighting = None
    best_residual_sum = float('inf')
    model_scores= []
        
    models_to_compare= ['Exponential','Spherical','Gaussian']
    weightings_to_compare= [None, 'linear','exp']  # None, 'linear','exp','sq'
    # dist_weighting= 'linear'; model_name= 'Exponential';
    
    scores= pd.DataFrame(columns=['Model Type','Distance Weighting','Residuals Sum',
                                  'Weighted Residuals Sum','RMSE'])
    
    for dist_weighting in weightings_to_compare:
        for model_name in models_to_compare:
            variogram = skg.Variogram(coords, values, bin_func= "even",
                                      n_lags= n_lags, maxlag= maxlag,
                                      fit_sigma= dist_weighting, # Dist. Weighting
                                      model= model_name,         # Model type
                                      normalize=False)
            # variogram.set_model(model_name)
            if bool(plot):
                fig= variogram.plot(show=False) # show=False
                fig.suptitle(f'{model_name} model [Dist. weighting: {dist_weighting} ]')
                fig.savefig(f'{outdir}Model_{model_name}_weight_{dist_weighting}.png',dpi=300)
                plt.close()
        
            residuals= variogram.residuals
            residual_sum = np.sum(residuals**2)
            
            if weighting== 'linear':
                residuals= np.arange(n_lags,0,-1) * (residuals ** 2) # Linear weighting
            # residuals= 1/np.exp(residuals**2)               # Exponential weighting (needs fixing)
            # residuals= np.sqrt(residuals ** 2)              # Square root weighting
            # 
                weighted_sum = np.sum(residuals)

            scores.loc[len(scores)]= [model_name,dist_weighting,residual_sum,
                                      weighted_sum, variogram.rmse]
            # print(f"Model: {model_name}; Weighting: {dist_weighting}; Res. Sum: {residual_sum}")
        
            if residual_sum < best_residual_sum:
                best_model = model_name
                best_weighting = dist_weighting
                best_residual_sum = residual_sum
                
    scores.to_csv(f'{outdir}model_scores.csv',index=False)
    
    return best_model, best_weighting


    
#-----------------------------------------------------------------------------
# Plot variograms ------------------------------------------------------------
#-----------------------------------------------------------------------------
def variogram_plot(exp_variograms,mod_variograms=None,plot_model=True):
    fig = plt.figure(figsize=(10, 7))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set1.colors)
    plt.rcParams["font.size"]= 14
    
    mfig, maxes= cfig.figure_setup(region,plot_features=['IceCatchments','GroundingLine','Stations'])
    
    for i, key in enumerate(exp_variograms.keys()):
        plt.figure(fig)  # activate figure
        p= plt.plot(exp_variograms[key].bins,
                    exp_variograms[key].experimental,
                    '.',alpha=0.6,label=f'Experimental: {key}')
        p_color= p[0].get_color()
            
        plt.plot([0,maxlag],[1.0,1.0],color = 'black')
        plt.xlabel(r'Lag Distance $\bf(h)$, (m)')
        plt.ylabel(r'$\gamma \bf(h)$')
        plt.title(f'Variograms: {n_clusters} clusters')
        plt.legend(loc='upper left',fontsize='x-small')
        plt.xlim([0,maxlag])
        plt.grid(True,linewidth=0.4,linestyle='--',alpha=0.6)
        
        if bool(plot_model):
            # plt.plot(lagh,mod_variograms[key][3],'-', 
            #          color= p_color, # plot same color as exp vario
            #          label= f'Modeled: {key}')
            plt.plot(exp_variograms[key].data()[0],
                     exp_variograms[key].data()[1],
                     '-', color= p_color, # plot same color as exp vario
                     label= f'Modeled: {key}')
        
        # Make map of k-means data points------------------------
        plt.figure(mfig)  # activate map figure
        # x,y = locals()[key].X, locals()[key].Y
        df= pd.read_csv(f'{outdir}exp_variograms/{key}_data.csv')
        x,y = df.X, df.Y
        plt.scatter(x,y, color= p_color, label=f'{key}',
                    marker=".", s= 2, edgecolors='none')
        plt.legend(loc='upper right',fontsize='xx-small',markerscale=1.5, 
                   scatterpoints= 3, ncol=2)
        
    if bool(plot_model):
        outname= 'exp_variogram_model_fit'
    else:
        outname= 'exp_variogram'
        
    fig.savefig(f'{outdir}{outname}.png',
                transparent=True,bbox_inches="tight",dpi= 600)
    mfig.savefig(outdir+'k-means_cluster_map.png',
                 transparent=True,bbox_inches="tight",dpi= 600)
    plt.close(); plt.close();
    



#-----------------------------------------------------------------------------
# Calculate experimental variogram -------------------------------------------
#-----------------------------------------------------------------------------
def experimental_variogram():
    df_samp= pd.read_parquet(outdir+'df_samp.parquet')
    df_samp= df_samp.sample(frac=downsample_fraction)
    
    df_samp.to_parquet(outdir+'df_variogram_calc.parquet')
    cfig.plot_data_points(df_samp,
                  f'{outdir}df_variogram_calc.png',region,
                  dmap={'X':'x','Y':'y','Bed':'z'},
                  title='Downsampled data for variogram calculation',cbar_label='Bed elevation (m)')
    
    # Dict for mapping variogram function
    exp_variograms= dict(zip([f'df{k}' for k in range(n_clusters)],
                          [v for v in range(n_clusters)]))
    
    for key, value in exp_variograms.items():
        locals()[key] = df_samp[df_samp['K'] == value] # isolate cluster 0 data
        
        coords = locals()[key][['X','Y']].values
        values = locals()[key]['Nbed']
        
        model_type, weighting = find_best_model(coords, values, n_lags, maxlag,
                                        f'{outdir}/model_fitting/{key}/',
                                        weighting='linear',plot=True)
        
        variogram = skg.Variogram(coords, values, bin_func= "even",
                                  n_lags= n_lags, maxlag= maxlag,
                                  normalize=False,
                                  model= model_type,
                                  fit_sigma=weighting)
        
        #--------------------------------------------------------
        # Write out experimental variogram bins + values to file
        #--------------------------------------------------------
        if not os.path.exists(exp_v_dir):
            os.makedirs(exp_v_dir)
            
        locals()[key].to_csv(f'{exp_v_dir}{key}_data.csv')
        
        pd.DataFrame({'bins':variogram.bins,
                      'experimental':variogram.experimental}).to_parquet(
            f'{exp_v_dir}exp_vario_{key}.parquet')
        
        pd.DataFrame(variogram.parameters).T.to_csv(
            f'{exp_v_dir}exp_params_{key}.csv',
            index=False,header=['range','sill','nugget'])
        
        #--------------------------------------------------------
        exp_variograms[key]= variogram  # overwrite key value with relevant variogram
             
        fprint(f'Experimental variogram for {key}')
        
    variogram_plot(exp_variograms,plot_model=False)
    
    return exp_variograms
    
    
#-----------------------------------------------------------------------------
# Fit variogram model --------------------------------------------------------
#-----------------------------------------------------------------------------
def fit_variogram_model(exp_variograms):
  
    df_gamma= pd.DataFrame()
    
    for key, vario in exp_variograms.items():
        model_type= vario.describe()['model']
        parameters= vario.parameters
        vrange=  parameters[0] # vario.describe()['effective_range'] # effective range from model vgram dict
        vsill=   parameters[1] # vario.describe()['sill'] 
        vnugget= parameters[2] 
        gam= [Azimuth, vnugget, vrange, vrange, vsill, model_type, key]
        
        df_gamma= pd.concat([df_gamma, pd.DataFrame(gam).T], axis=0,ignore_index=True)
        
    df_gamma.to_csv(f'{outdir}vario.txt',index=False,header=False)
    variogram_plot(exp_variograms,plot_model=True)
    fprint('Saved + plotted variograms')
 
    # return mod_variograms
 
    
#-----------------------------------------------------------------------------
# Run main functions
#-----------------------------------------------------------------------------
def main():
    if bool(import_data_norm_resample_cluster):
        data_preprocessing()
    
    if bool(calc_experimental_variogram):
        exp_variograms= experimental_variogram()
    
    if bool(anisotropic_variogram):
        exp_variograms= exp_anisotropic_variogram()
    
    if bool(fit_model_to_variogram):
        fit_variogram_model(exp_variograms)
        
    return exp_variograms
        
if __name__ == "__main__":
    
    exp_variograms= main()
    
#-----------------------------------------------------------------------------
        
      

#-----------------------------------------------------------------------------
# Anisotropic experimental variogram -----------------------------------------
#-----------------------------------------------------------------------------
def exp_anisotropic_variogram():
    df_samp= pd.read_parquet(outdir+'df_samp.parquet')
    df_samp= df_samp.sample(frac=downsample_fraction)
    
    df_samp.to_parquet(outdir+'df_variogram_calc.parquet')
    cfig.plot_data_points(df_samp,
                  f'{outdir}df_variogram_calc.png',region,
                  dmap={'X':'x','Y':'y','Bed':'z'},
                  title='Downsampled data for variogram calculation',cbar_label='Bed elevation (m)')
    
    # lag_dist = lag_dist  # calc. n_lags from this if provided
    # maxlag = max_lag
    n_lags= int(maxlag/lag_dist)
    # n_lags = 60     # num of bins
    
    # dict for mapping variogram function
    exp_variograms= dict(zip([f'df{k}' for k in range(n_clusters)],
                          [v for v in range(n_clusters)]))
    
    for key, value in exp_variograms.items():
        locals()[key] = df_samp[df_samp['K'] == value] # isolate cluster 0 data
        coords = locals()[key][['X','Y']].values
        values = locals()[key]['Nbed']
        
        # Directional variograms
        V0 = skg.DirectionalVariogram(coords, values, bin_func = "even", n_lags = n_lags, 
                                          maxlag = maxlag, normalize=False, azimuth=0, tolerance=15)
        V45 = skg.DirectionalVariogram(coords, values, bin_func = "even", n_lags = n_lags, 
                                          maxlag = maxlag, normalize=False, azimuth=45, tolerance=15)
        V90 = skg.DirectionalVariogram(coords, values, bin_func = "even", n_lags = n_lags, 
                                           maxlag = maxlag, normalize=False, azimuth=90, tolerance=15)
        V135 = skg.DirectionalVariogram(coords, values, bin_func = "even", n_lags = n_lags, 
                                          maxlag = maxlag, normalize=False, azimuth=135, tolerance=15)
        # Extract parameters 
        x0 = V0.bins
        y0 = V0.experimental
        x45 = V45.bins
        y45 = V45.experimental
        x90 = V90.bins
        y90 = V90.experimental
        x135 = V135.bins
        y135 = V135.experimental
        
        # plot multidirectional variogram
        plt.figure(figsize=(6,4))
        plt.scatter(x0, y0, s=12, label='0 degrees (East-West)')
        plt.scatter(x45, y45,s=12, c='g', label='45 degrees')
        plt.scatter(x90, y90, s=12, c='r', label='90 degrees')
        plt.scatter(x135, y135, s=12, c='k', label='135 degrees')
        plt.title(f'{key}: Anisotropoic experimental variogram')
        plt.xlabel('Lag [m]'); plt.ylabel('Semivariance')
        plt.legend(loc='lower right',fontsize='x-small')
        plt.show()
        plt.savefig(f'{outdir}exp_variogram_{key}.png',
                transparent=True,bbox_inches="tight",dpi= 600)
        plt.close()

    # # dict for mapping variogram function
    # exp_variograms= dict(zip([f'df{k}' for k in range(n_clusters)],
    #                       [v for v in range(n_clusters)]))
    
    # for key, value in exp_variograms.items():
    #     locals()[key] = df_samp[df_samp['K'] == value] # isolate cluster 0 data
    #     coords = locals()[key][['X','Y']].values
    #     values = locals()[key]['Nbed']
    #     variogram = skg.Variogram(coords, values, bin_func= "even",
    #                               model= 'exponential', n_lags= n_lags, 
    #                        maxlag= maxlag, normalize=False)
    #     exp_v_dir= f'{outdir}exp_variograms/'
    #     if not os.path.exists(exp_v_dir):
    #         os.makedirs(exp_v_dir)
    #     variogram.to_DataFrame().to_parquet(
    #         f'{exp_v_dir}exp_vario_{key}.parquet')
        
    #     exp_variograms[key]= variogram  # overwrite key value with relevant variogram
        
    #     print(f'Exp. variogram for {key} calculated...........')
    
    return exp_variograms
        
        
        
        