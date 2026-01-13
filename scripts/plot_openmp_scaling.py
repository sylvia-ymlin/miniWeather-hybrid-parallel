#!/usr/bin/env python3
"""
Plot OpenMP and Hybrid MPI+OpenMP scaling results with unified visual style.
Color scheme: Pure MPI (Red), Pure OpenMP (Blue), Hybrid (Green)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Unified Color Scheme
# ============================================================================
COLOR_MPI = '#E74C3C'       # Red for Pure MPI
COLOR_OPENMP = '#2E86AB'    # Blue for Pure OpenMP  
COLOR_HYBRID = '#27AE60'    # Green for Hybrid
COLOR_IDEAL = '#7F8C8D'     # Gray for ideal lines

# ============================================================================
# Data from Intel Xeon Platinum 8358P Testing (15 vCPU, AutoDL Cloud)
# Problem size: 400x200 grid, 10s simulation
# ============================================================================

# OpenMP Thread Scaling Data
omp_threads = np.array([1, 2, 4, 8, 12])
omp_times = np.array([6.357, 3.302, 1.795, 0.891, 0.670])
omp_speedup = omp_times[0] / omp_times
omp_efficiency = (omp_speedup / omp_threads) * 100.0

# Hybrid MPI+OpenMP Data (8 total cores)
hybrid_configs = ['Pure OpenMP\n(8 threads)', 'Hybrid MPI 2×4\n(2 Ranks, 4 Threads)', 'Hybrid MPI 4×2\n(4 Ranks, 2 Threads)']
hybrid_times = np.array([0.891, 0.872, 1.009])

# ============================================================================
# Create Figure
# ============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# --- Plot 1: OpenMP Speedup (Log-Log for better visualization) ---
ax1 = axs[0]
ax1.loglog(omp_threads, omp_speedup, 'o-', linewidth=2.5, markersize=10, 
           color=COLOR_OPENMP, label='Measured Speedup', zorder=3)
ax1.loglog(omp_threads, omp_threads, '--', linewidth=2, color=COLOR_IDEAL, 
           alpha=0.8, label='Ideal (Linear)', zorder=2)
ax1.fill_between(omp_threads, omp_speedup, alpha=0.2, color=COLOR_OPENMP)

ax1.set_xlabel('OpenMP Threads', fontsize=12, fontweight='bold')
ax1.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
ax1.set_title('OpenMP Thread Scaling\n(400×200 grid, Intel Xeon Platinum)', 
              fontsize=13, fontweight='bold')
ax1.set_xticks(omp_threads)
ax1.set_xticklabels(omp_threads)
ax1.set_yticks([1, 2, 4, 8, 12])
ax1.set_yticklabels(['1×', '2×', '4×', '8×', '12×'])
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, which='both', linestyle=':', alpha=0.5)
ax1.set_xlim(0.8, 15)
ax1.set_ylim(0.8, 15)

# Annotate speedup values
for t, s in zip(omp_threads, omp_speedup):
    ax1.annotate(f'{s:.2f}×', (t, s), textcoords="offset points", 
                 xytext=(8, 5), ha='left', fontweight='bold', fontsize=10)

# --- Plot 2: OpenMP Efficiency ---
ax2 = axs[1]
bar_colors = [COLOR_OPENMP if e >= 80 else '#F39C12' for e in omp_efficiency]
bars = ax2.bar(omp_threads, omp_efficiency, color=bar_colors, 
               edgecolor='black', linewidth=1.2, width=0.8)
ax2.axhline(y=100, color=COLOR_IDEAL, linestyle='--', linewidth=2, 
            alpha=0.8, label='Ideal (100%)')
ax2.axhline(y=80, color=COLOR_HYBRID, linestyle=':', linewidth=1.5, 
            alpha=0.8, label='Good Threshold (80%)')

ax2.set_xlabel('OpenMP Threads', fontsize=12, fontweight='bold')
ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
ax2.set_title('OpenMP Parallel Efficiency', fontsize=13, fontweight='bold')
ax2.set_xticks(omp_threads)
ax2.set_ylim(0, 115)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, axis='y', linestyle=':', alpha=0.5)

# Annotate efficiency values (unified font size)
for bar, eff in zip(bars, omp_efficiency):
    ax2.annotate(f'{eff:.1f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                 textcoords="offset points", xytext=(0, 5), ha='center', 
                 fontweight='bold', fontsize=10)

# --- Plot 3: Hybrid MPI+OpenMP Comparison ---
ax3 = axs[2]
x_pos = np.arange(len(hybrid_configs))
colors3 = [COLOR_OPENMP, COLOR_HYBRID, COLOR_MPI]
bars3 = ax3.bar(x_pos, hybrid_times, color=colors3, edgecolor='black', linewidth=1.2)

ax3.set_xlabel('Configuration (8 total cores)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
ax3.set_title('Hybrid MPI+OpenMP Comparison\n(Same Total Parallelism)', 
              fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(hybrid_configs, fontsize=9)
ax3.set_ylim(0, 1.3)
ax3.grid(True, axis='y', linestyle=':', alpha=0.5)

# Annotate with time and relative performance
annotations = ['Baseline', 'Best: +2.1%', 'Worst: -11.7%']
for bar, time, ann in zip(bars3, hybrid_times, annotations):
    ax3.annotate(f'{time:.3f}s\n{ann}', 
                 (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03),
                 ha='center', fontsize=10, fontweight='bold')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_OPENMP, edgecolor='black', label='Pure OpenMP'),
    Patch(facecolor=COLOR_HYBRID, edgecolor='black', label='Hybrid (Optimal)'),
    Patch(facecolor=COLOR_MPI, edgecolor='black', label='More MPI Ranks')
]
ax3.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('docs/openmp_scaling_results.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Plot saved to docs/openmp_scaling_results.png")
