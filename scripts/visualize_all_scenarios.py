#!/usr/bin/env python3
"""
Visualize all miniWeather simulation scenarios.
Generates evolution plots for each scenario and a summary comparison.
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os

# Change to the directory containing the output files
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

scenarios = [
    ('output_collision.nc', 'Colliding Thermals', 'collision'),
    ('output_thermal.nc', 'Rising Thermal Bubble', 'thermal'),
    ('output_gravity.nc', 'Internal Gravity Waves', 'gravity'),
    ('output_density.nc', 'Density Current (Cold Front)', 'density')
]

# Generate individual evolution plots for each scenario
for filename, title, name in scenarios:
    if not os.path.exists(filename):
        print(f"⚠️  跳过 {filename} (文件不存在)")
        continue
    try:
        data = nc.Dataset(filename)
        theta = data.variables['theta'][:]
        t = data.variables['t'][:]
        
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        timesteps = [0, len(t)//3, 2*len(t)//3, -1]
        
        for ax, ts in zip(axes, timesteps):
            im = ax.contourf(theta[ts, :, :], levels=50, cmap='RdBu_r')
            ax.set_title(f't = {t[ts]:.1f}s', fontsize=11)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_aspect('equal')
        
        plt.suptitle(f'{title} - miniWeather Simulation', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=axes, label='θ perturbation (K)', shrink=0.8)
        plt.tight_layout()
        
        output_file = f'docs/{name}_evolution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {output_file}")
        data.close()
    except Exception as e:
        print(f"❌ {filename}: {e}")

# Generate summary comparison plot (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

valid_count = 0
for ax, (filename, title, name) in zip(axes, scenarios):
    if not os.path.exists(filename):
        ax.text(0.5, 0.5, f'File not found:\n{filename}', ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        continue
    try:
        data = nc.Dataset(filename)
        theta = data.variables['theta'][:]
        im = ax.contourf(theta[-1, :, :], levels=50, cmap='RdBu_r')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label='θ (K)', shrink=0.8)
        data.close()
        valid_count += 1
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
        ax.set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('miniWeather: All Simulation Scenarios (Final State)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/all_scenarios.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 保存: docs/all_scenarios.png")

print(f"\n=== 完成 ({valid_count}/4 场景) ===")

