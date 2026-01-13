#!/usr/bin/env python3
"""Debug script to check NetCDF file structure and data values."""
import netCDF4 as nc
import numpy as np
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'output_collision_long.nc'

try:
    data = nc.Dataset(filename)
    print(f"File: {filename}")
    print(f"Variables: {list(data.variables.keys())}")
    
    theta = data.variables['theta']
    print(f"\ntheta dimensions: {theta.dimensions}")
    print(f"theta shape: {theta.shape}")
    print(f"theta size: {theta.size}")
    
    theta_data = theta[:]
    print(f"\ntheta_data shape: {theta_data.shape}")
    print(f"theta_data dtype: {theta_data.dtype}")
    
    print(f"\nFirst timestep data:")
    ts0 = theta_data[0] if len(theta_data.shape) == 3 else theta_data
    print(f"  Shape: {ts0.shape}")
    print(f"  Min: {ts0.min():.6e}")
    print(f"  Max: {ts0.max():.6e}")
    print(f"  Mean: {ts0.mean():.6e}")
    print(f"  Std: {ts0.std():.6e}")
    print(f"  Non-zero count: {np.count_nonzero(ts0)}")
    
    t = data.variables['t'][:]
    print(f"\nTime values: {t}")
    
    data.close()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

