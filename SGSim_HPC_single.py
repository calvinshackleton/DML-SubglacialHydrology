#!/usr/bin/env python
#------------------------------------------------------------------------------
"""
Gridding and interpolating using Sequential Gaussian Simulation
    - Simulates realistically rough topography between measurements using spatial statistics of nearby data
    - Designed to run in a High Performace Computing (HPC) environment with $numsim seperate tasks (single process not multiprocess)
    - Requires: optimum variogram parameters determined by experimenting first

Created on 28-03-2023
@author: calvin.shackleton@npolar.no
Some functions used/modified from: https://github.com/GatorGlaciology/GStatSim/blob/main/demos/7_non-stationary_SGS_example1.ipynb

"""
#------------------------------------------------------------------------------
import pandas as pd
import os, sys
import gstatsim_DML as gs
import create_figs_DML as cfig
import utility_DML as util
# import matplotlib
# matplotlib.use('Agg') # Background rendering
#import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import pyproj
# import numpy as np
from rasterio.crs import CRS


"""-------------------------------------------------------------------------------
# Define workflow loop to run simulation functions [Set options below]
-------------------------------------------------------------------------------"""
def SGSim_loop(i):
    print(f"\n--------- Simulation {i} of {istart+nsim} ------------------------")
    folder_m= folder+ str(i).zfill(3) + '/'
    if not os.path.exists(folder_m):
        os.makedirs(folder_m)
        
    # Define coordinate grid------------------------------------------------------
    xmin, xmax, ymin, ymax = [v for v in region]
    Pred_grid_xy= util.pred_grid(xmin, xmax, ymin, ymax, res)
    
    df_samp= pd.read_parquet(indata_path)
    
    # Normal score transformation-------------------------------------------------
    data = df_samp['Bed'].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
    df_samp['Nbed_'] = nst_trans.transform(data)
    
    # SGSim cluster method--------------------------------------------------------
    maskdf= pd.read_parquet(f'{datadir}terrestrial_only_xy_{res}m.parquet')
    sgsim = gs.Interpolation.cluster_sgs(Pred_grid_xy, df_samp,
                                         'X', 'Y', 'Nbed_', 'K',
                                         k, df_gamma, rad, maskdf) 
    del df_samp, maskdf

    #---------------------------------------------------------------------
    # Backtransform function, convert to grid, save xyz + geotiff
    #---------------------------------------------------------------------
    sgs = sgsim.reshape(-1,1) 
    sgs_trans = nst_trans.inverse_transform(sgs) # reverse normal score transformation

    d={'x':Pred_grid_xy[:,0],'y':Pred_grid_xy[:,1],'z':sgs_trans.squeeze(),'nBed':sgs.squeeze()}
    xyz_points= pd.DataFrame(data=d)
    
    xyz_points.to_parquet(folder_m+'SGSim.parquet') # save points
    
    # xyz_points= pd.read_parquet(folder_m+'SGSim.parquet') # read output for troubleshooting
    grid,xx,yy= util.points_to_grid(xyz_points.x, xyz_points.y,
                                   xyz_points.z, region, res) 
    util.write_geotiff(folder_m+'SGSim.tif',xx,yy,grid,region) # save geotiff

    
    # Plot results----------------------------------------------------------------
    cfig.topo_figure(folder_m+'SGSim.tif',folder_m+'Bed.png',
                     'Bed Elevation (m)',cmap='viridis',contours=False)
  



"""-------------------------------------------------------------------------------
 Set options and parameters for simulations and run functions
-------------------------------------------------------------------------------"""

# Options-----------------------------
areatag= 'DML'  # east or west
vgram= f'{areatag}_v2.5'  # Variogram model version
outtag= f'{areatag}_v3.4' # Output version

# Set region of interest [x1,x2,y1,y2]----------------------------------------
target_crs = pyproj.CRS(CRS.from_dict({'proj':"stere",'lat_ts': -71,'units': 'm',
                                      'lat_0':-90, 'lon_0':10, 'ellps': 'WGS84'}))
    
region= [-700000,1300000,1800000,2230000] # Custom CRS full region

#-----------------------------------------------------------------------------
run_in_parallel= False
only_terrestrial_regions= True

clip_shapefile= '/cluster/home/cshack/data/RINGS_workshop_grounded_2022.shp'
clip_buffer= 1000

istart= int(sys.argv[1]) # Simulation number        *set in HPC submit script*
nsim=   int(sys.argv[2]) # Number of realizations   *set in HPC submit script*

res= 1000   # Out grid resolution (m)
rad= 200000 # Search radius (m) 
k=   50     # Max. no. neighboring data points

indir=       '/cluster/.../' # Set relevant location on HPC
datadir=     '/cluster/.../' # Set relevant location on HPC
vgram_dir=   f'{indir}data/variogram/{vgram}/' 
indata_path= f'{vgram_dir}df_samp.parquet'  

folder= f'/cluster/.../{outtag}/' # Set relevant location on HPC

if not os.path.exists(folder):
    os.makedirs(folder)
    
# Read in data and variogram parameters---------------------------------------
vario= pd.read_csv(vgram_dir+'vario.txt',header=None) # Variogram model
vario= [list(a) for a in vario.values] # as list 

df_gamma = pd.DataFrame({'Variogram': [row for row in vario]})

SGSim_loop(istart)
#------------------------------------------------------------------------------

