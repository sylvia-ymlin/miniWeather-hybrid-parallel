#!/usr/bin/env python3
"""Professional visualization for miniWeather simulations."""

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.style.use('default')
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['font.size'] = 12

def plot_single(theta_2d, title, filename, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if vmin is None:
        vmax = max(abs(theta_2d.min()), abs(theta_2d.max()))
        vmin = -vmax
    
    im = ax.contourf(theta_2d, levels=100, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_xlabel('x (grid points)', fontsize=14)
    ax.set_ylabel('z (grid points)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Potential Temperature Perturbation (K)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {filename}')

# Rising Thermal
try:
    data = nc.Dataset('output_thermal_long.nc')
    theta = data.variables['theta'][:]
    t = data.variables['t'][:]
    for ts in [5, -1]:
        time_val = int(t[ts])
        plot_single(theta[ts], f'Rising Thermal - {time_val}s', f'docs/thermal_{time_val}s.png')
    data.close()
except Exception as e:
    print(f'Thermal error: {e}')

# Colliding Thermals
try:
    data = nc.Dataset('output_collision_long.nc')
    theta = data.variables['theta'][:]
    t = data.variables['t'][:]
    for ts in [2, 4, -1]:
        time_val = int(t[ts])
        plot_single(theta[ts], f'Colliding Thermals - {time_val}s', f'docs/collision_{time_val}s.png')
    data.close()
except Exception as e:
    print(f'Collision error: {e}')

# Density Current
try:
    data = nc.Dataset('output_density_long.nc')
    theta = data.variables['theta'][:]
    t = data.variables['t'][:]
    for ts in [2, -1]:
        time_val = int(t[ts])
        plot_single(theta[ts], f'Density Current - {time_val}s', f'docs/density_{time_val}s.png')
    data.close()
except Exception as e:
    print(f'Density error: {e}')

# Mountain Gravity Waves
try:
    data = nc.Dataset('output_gravity_long.nc')
    theta = data.variables['theta'][:]
    t = data.variables['t'][:]
    for ts in [4, -1]:
        time_val = int(t[ts])
        plot_single(theta[ts], f'Mountain Gravity Waves - {time_val}s', f'docs/gravity_{time_val}s.png')
    data.close()
except Exception as e:
    print(f'Gravity Waves error: {e}')

# Injection
try:
    data = nc.Dataset('output_injection_long.nc')
    theta = data.variables['theta'][:]
    t = data.variables['t'][:]
    for ts in [3, -1]:
        time_val = int(t[ts])
        plot_single(theta[ts], f'Injection - {time_val}s', f'docs/injection_{time_val}s.png')
    data.close()
except Exception as e:
    print(f'Injection error: {e}')

print('\n=== Done ===')

