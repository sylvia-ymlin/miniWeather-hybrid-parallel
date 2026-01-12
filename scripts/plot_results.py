import matplotlib.pyplot as plt
import numpy as np

# Data from PROJECT_ANALYSIS.md
ranks = np.array([1, 2, 3, 4])

# Strong Scaling Data (100x50 fixed problem size)
strong_times = np.array([4.95, 2.52, 1.92, 1.76])
strong_speedup = strong_times[0] / strong_times
strong_ideal_speedup = ranks
strong_efficiency = (strong_speedup / ranks) * 100.0

# Weak Scaling Data (100x50 per rank)
weak_times = np.array([12.70, 25.03, 23.56, 38.31])
weak_efficiency = (weak_times[0] / weak_times) * 100.0

# Setup 2x2 Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- Strong Scaling ---
# 1. Time (Log-Log)
axs[0, 0].plot(ranks, strong_times, 'o-', linewidth=2, color='#1f77b4', label='Measured Time')
axs[0, 0].plot(ranks, strong_times[0] / ranks, '--', color='gray', alpha=0.7, label='Ideal (Linear Speedup)')
axs[0, 0].set_title('Strong Scaling: Time to Solution\n(Fixed Problem Size 100x50)', fontsize=12, fontweight='bold')
axs[0, 0].set_ylabel('Time (s)', fontsize=10)
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_yscale('log', base=10)
axs[0, 0].set_xticks(ranks)
axs[0, 0].set_xticklabels(ranks)
axs[0, 0].grid(True, linestyle=':', alpha=0.7)
axs[0, 0].legend()

# 2. Efficiency
axs[0, 1].plot(ranks, strong_efficiency, 'o-', linewidth=2, color='#1f77b4', label='Efficiency')
axs[0, 1].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal')
axs[0, 1].set_title('Strong Scaling: Parallel Efficiency', fontsize=12, fontweight='bold')
axs[0, 1].set_ylabel('Efficiency (%)', fontsize=10)
axs[0, 1].set_ylim(0, 110)
axs[0, 1].set_xticks(ranks)
axs[0, 1].grid(True, linestyle=':', alpha=0.7)
for i, txt in enumerate(strong_efficiency):
    axs[0, 1].annotate(f"{txt:.1f}%", (ranks[i], strong_efficiency[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

# --- Weak Scaling ---
# 3. Time (Linear)
axs[1, 0].plot(ranks, weak_times, 's-', linewidth=2, color='#ff7f0e', label='Measured Time')
axs[1, 0].axhline(y=weak_times[0], color='gray', linestyle='--', alpha=0.7, label='Ideal (Constant Time)')
axs[1, 0].set_title('Weak Scaling: Time to Solution\n(Fixed Work per Rank 100x50)', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('MPI Ranks', fontsize=10)
axs[1, 0].set_ylabel('Time (s)', fontsize=10)
axs[1, 0].set_ylim(0, max(weak_times)*1.2)
axs[1, 0].set_xticks(ranks)
axs[1, 0].grid(True, linestyle=':', alpha=0.7)
axs[1, 0].legend()

# 4. Efficiency
axs[1, 1].plot(ranks, weak_efficiency, 's-', linewidth=2, color='#ff7f0e', label='Efficiency')
axs[1, 1].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal')
axs[1, 1].set_title('Weak Scaling: Parallel Efficiency', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('MPI Ranks', fontsize=10)
axs[1, 1].set_ylabel('Efficiency (%)', fontsize=10)
axs[1, 1].set_ylim(0, 110)
axs[1, 1].set_xticks(ranks)
axs[1, 1].grid(True, linestyle=':', alpha=0.7)
for i, txt in enumerate(weak_efficiency):
    axs[1, 1].annotate(f"{txt:.1f}%", (ranks[i], weak_efficiency[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('docs/scaling_results.png', dpi=300)
print("Plot saved to scaling_results.png")
