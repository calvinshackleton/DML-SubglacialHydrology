#!/usr/bin/env python
# coding: utf-8
"""
Functions for geostatistical simulation

    - Uses a sequential Gaussian simulation algorithm that uses different variogram model for different data clusters 
    - Requires: Pre-clustered data and modelled variograms (see exp_variogram*.py)
    
    *Code modified/based on GStatSim package: https://github.com/GatorGlaciology/GStatSim/blob/main/gstatsim.py
"""

import numpy as np
import numpy.linalg as linalg
import pandas as pd
import sklearn as sklearn
from sklearn.neighbors import KDTree
import math
from scipy.spatial import distance_matrix
from scipy.interpolate import Rbf
from tqdm import tqdm
import random
from sklearn.metrics import pairwise_distances

###################################

# RBF trend estimation

###################################

def rbf_trend(grid_matrix, smooth_factor, res):
    """
    Estimate trend using radial basis functions
    
    Parameters
    ----------
        grid_matrix : numpy.ndarray
            matrix of gridded conditioning data
        smooth_factor : float
            Parameter controlling smoothness of trend. Values greater than 
            zero increase the smoothness of the approximation.
        res : float
            grid cell resolution
            
    Returns
    -------
        trend_rbf : numpy.ndarray
            RBF trend estimate
    """ 
    sigma = np.rint(smooth_factor/res)
    ny, nx = grid_matrix.shape
    rbfi = Rbf(np.where(~np.isnan(grid_matrix))[1],
               np.where(~np.isnan(grid_matrix))[0], 
               grid_matrix[~np.isnan(grid_matrix)],smooth = sigma)

    # evaluate RBF
    yi = np.arange(nx)
    xi = np.arange(ny)
    xi,yi = np.meshgrid(xi, yi)
    trend_rbf = rbfi(xi, yi)   
    
    return trend_rbf


####################################

# Nearest neighbor octant search

####################################

class NearestNeighbor:

    def center(arrayx, arrayy, centerx, centery):
        """
        Shift data points so that grid cell of interest is at the origin
        
        Parameters
        ----------
            arrayx : numpy.ndarray
                x coordinates of data
            arrayy : numpy.ndarray
                y coordinates of data
            centerx : float
                x coordinate of grid cell of interest
            centery : float
                y coordinate of grid cell of interest
        
        Returns
        -------
            centered_array : numpy.ndarray
                array of coordinates that are shifted with respect to grid cell of interest
        """ 
        
        centerx = arrayx - centerx
        centery = arrayy - centery
        centered_array = np.array([centerx, centery])
        
        return centered_array

    def distance_calculator(centered_array):
        """
        Compute distances between coordinates and the origin
        
        Parameters
        ----------
            centered_array : numpy.ndarray
                array of coordinates
        
        Returns
        -------
            dist : numpy.ndarray
                array of distances between coordinates and origin
        """ 
        
        dist = np.linalg.norm(centered_array, axis=0)
        
        return dist

    def angle_calculator(centered_array):
        """
        Compute angles between coordinates and the origin
        
        Parameters
        ----------
            centered_array : numpy.ndarray
                array of coordinates
        
        Returns
        -------
            angles : numpy.ndarray
                array of angles between coordinates and origin
        """ 
        
        angles = np.arctan2(centered_array[0], centered_array[1])
        
        return angles

    
    def nearest_neighbor_search_cluster(radius, num_points, loc, data2):
        """
        Nearest neighbor octant search when doing sgs with clusters
        
        Parameters
        ----------
            radius : int, float
                search radius
            num_points : int
                number of points to search for
            loc : numpy.ndarray
                coordinates for grid cell of interest
            data2 : pandas DataFrame
                data 
        
        Returns
        -------
            near : numpy.ndarray
                nearest neighbors
            cluster_number : int
                nearest neighbor cluster number
        """ 
        
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array)
        data["angles"] = NearestNeighbor.angle_calculator(centered_array)
        data = data[data.dist < radius] 
        
        data = data.sort_values('dist', ascending = True).dropna() # added dropna() to remove nans at ocean cells
        data = data.reset_index() 
        cluster_number = data.K[0] 

        bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, 
                math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
        data["Oct"] = pd.cut(data.angles, bins = bins, labels = list(range(8))) 
        oct_count = num_points // 8
        smallest = np.ones(shape=(num_points, 3)) * np.nan

        for i in range(8):
            octant = data[data.Oct == i].iloc[:oct_count][['X','Y','Z']].values
            for j, row in enumerate(octant):
                smallest[i*oct_count+j,:] = row 

        near = smallest[~np.isnan(smallest).any(axis=1)].reshape(-1,3)

        return near, cluster_number



#########################

# Rotation Matrix

#########################

def make_rotation_matrix(azimuth, major_range, minor_range):
    """
    Make rotation matrix for accommodating anisotropy
    
    Parameters
    ----------
        azimuth : int, float
            angle (in degrees from horizontal) of axis of orientation
        major_range : int, float
            range parameter of variogram in major direction, or azimuth
        minor_range : int, float
            range parameter of variogram in minor direction, or orthogonal to azimuth
    
    Returns
    -------
        rotation_matrix : numpy.ndarray
            2x2 rotation matrix used to perform coordinate transformations
    """
    
    theta = (azimuth / 180.0) * np.pi 
    
    rotation_matrix = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],]),
        np.array([[1 / major_range, 0], [0, 1 / minor_range]]))
    
    return rotation_matrix


###########################

# Covariance functions

###########################

class Covariance:
    
    def covar(effective_lag, sill, nug, vtype):
        """
        Compute covariance
        
        Parameters
        ----------
            effective_lag : int, float
                lag distance that is normalized to a range of 1
            sill : int, float
                sill of variogram
            nug : int, float
                nugget of variogram
            vtype : string
                type of variogram model (Exponential, Gaussian, or Spherical)
        
        Returns
        -------
            c : numpy.ndarray
                covariance
        """
        
        if vtype == 'Exponential' or 'exponential':
            c = (sill - nug)*np.exp(-3 * effective_lag)
        elif vtype == 'Gaussian' or 'gaussian':
            c = (sill - nug)*np.exp(-3 * np.square(effective_lag))
        elif vtype == 'Spherical' or 'spherical':
            c = sill - nug - 1.5 * effective_lag + 0.5 * np.power(effective_lag, 3)
            c[effective_lag > 1] = sill - 1
        return c

    def make_covariance_matrix(coord, vario, rotation_matrix):
        """
        Make covariance matrix showing covariances between each pair of input coordinates
        
        Parameters
        ----------
            coord : numpy.ndarray
                coordinates of data points
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            rotation_matrix : numpy.ndarray
                rotation matrix used to perform coordinate transformations
        
        Returns
        -------
            covariance_matrix : numpy.ndarray 
                nxn matrix of covariance between n points
        """
        
        nug = vario[1]
        sill = vario[4]  
        vtype = vario[5]
        mat = np.matmul(coord, rotation_matrix)
        effective_lag = pairwise_distances(mat,mat) 
        covariance_matrix = Covariance.covar(effective_lag, sill, nug, vtype) 

        return covariance_matrix

    def make_covariance_array(coord1, coord2, vario, rotation_matrix):
        """
        Make covariance array showing covariances between each data points and grid cell of interest
        
        Parameters
        ----------
            coord1 : numpy.ndarray
                coordinates of n data points
            coord2 : numpy.ndarray
                coordinates of grid cell of interest (i.e. grid cell being simulated) that is repeated n times
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            rotation_matrix - rotation matrix used to perform coordinate transformations
        
        Returns
        -------
            covariance_array : numpy.ndarray
                nx1 array of covariance between n points and grid cell of interest
        """
        
        nug = vario[1]
        sill = vario[4]
        vtype = vario[5]
        mat1 = np.matmul(coord1, rotation_matrix) 
        mat2 = np.matmul(coord2.reshape(-1,2), rotation_matrix) 
        effective_lag = np.sqrt(np.square(mat1 - mat2).sum(axis=1))
        covariance_array = Covariance.covar(effective_lag, sill, nug, vtype)

        return covariance_array
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
# Interpolation function
#-----------------------------------------------------------------------------
class Interpolation:

    def cluster_sgs(prediction_grid, df, xx, yy, zz, kk, num_points, df_gamma, radius, maskdf=None):
        """
        Sequential Gaussian simulation where variogram parameters are different for each k cluster. Uses simple kriging 
        
        Parameters
        ----------
            prediction_grid : numpy.ndarray
                x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df : pandas DataFrame
                data frame of conditioning data
            xx : string
                column name for x coordinates of input data frame
            yy : string 
                column name for y coordinates of input data frame
            zz : string
                column for z values (or data variable) of input data frame
            kk : string
                column of k cluster numbers for each point
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            radius : int, float
                search radius
            maskdf : added option to skip cells that fall outside of the mask
        
        Returns
        -------
            sgs : numpy.ndarray
                simulated value for each coordinate in prediction_grid
        """

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z", kk: "K"})  
        xyindex = np.arange(len(prediction_grid)) 
        random.shuffle(xyindex)
        mean_1 = np.average(df["Z"].values) 
        sgs = np.zeros(shape=len(prediction_grid)) 
        
        checklist=[tuple(row) for row in maskdf.to_numpy().tolist()] # Create checklist of coords to simulate



        for idx, predxy in enumerate(tqdm(prediction_grid, position=0,
                                      mininterval=28800, # maxinterval=86400,
                                      miniters=100000,
                                      leave=True)):
            z = xyindex[idx] 
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            
            if maskdf is not None:
                coord=prediction_grid[z:z+1,:]
                coord= [tuple(row) for row in coord.tolist()]
                
                if coord[0] in checklist:
                    if np.sum(test_idx==2)==0: 
                        
                        # gather nearest neighbor points and K cluster value
                        nearest, cluster_number = NearestNeighbor.nearest_neighbor_search_cluster(radius, 
                                                                                                  num_points, 
                                                                                                  prediction_grid[z],
                                                                                                  df[['X','Y','Z','K']])  
                        vario = df_gamma.Variogram[cluster_number] 
                        norm_data_val = nearest[:,-1]   
                        xy_val = nearest[:,:-1]   
        
                        # unpack variogram parameters
                        azimuth = vario[0]
                        major_range = vario[2]
                        minor_range = vario[3]
                        var_1 = vario[4]
                        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 
                        new_num_pts = len(nearest)
        
                        # covariance between data
                        covariance_matrix = np.zeros(shape=((new_num_pts, new_num_pts))) 
                        covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                           vario, 
                                                                                                           rotation_matrix) 
                        
                        # covariance between data and unknown
                        covariance_array = np.zeros(shape=(new_num_pts)) 
                        k_weights = np.zeros(shape=(new_num_pts))
                        covariance_array = Covariance.make_covariance_array(xy_val, np.tile(prediction_grid[z], new_num_pts), 
                                                                            vario, rotation_matrix)
                        covariance_matrix.reshape(((new_num_pts)), ((new_num_pts)))
                        k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) 
                        est = mean_1 + np.sum(k_weights*(norm_data_val - mean_1)) 
                        var = var_1 - np.sum(k_weights*covariance_array)
                        var = np.absolute(var) 
        
                        sgs[z] = np.random.normal(est,math.sqrt(var),1) 
                    else:
                        
                        sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
                        cluster_number = df['K'].values[np.where(test_idx==2)[0][0]]
                else:
                    sgs[z]= np.nan # -9999
                    cluster_number = np.nan # -9999

            coords = prediction_grid[z:z+1,:] 
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 
                                             'Z': [sgs[z]], 'K': [cluster_number]})], sort=False)

        return sgs


__all__ = ['NearestNeighbor', 'Covariance', 'Interpolation', 'rbf_trend','make_rotation_matrix']

def __dir__():
    return __all__

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__} has no attribute {name}')
    return globals()[name]