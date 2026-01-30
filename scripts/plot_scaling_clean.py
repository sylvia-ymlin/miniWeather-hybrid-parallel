import matplotlib.pyplot as plt
import numpy as np

# Data from Apple M3 benchmarking (Jan 2026)
threads = np.array([1, 2, 4, 8])
mean_times = np.array([0.9132, 0.4680, 0.2407, 0.1270])
std_devs = np.array([0.0619, 0.0021, 0.0059, 0.0094])

# Calculate Speedup
# Speedup = T_1 / T_N
# Propagate Error for Speedup: dS = S * sqrt((dt1/t1)^2 + (dtn/tn)^2)
base_time = mean_times[0]
base_err = std_devs[0]
speedup = base_time / mean_times
speedup_err = speedup * np.sqrt((base_err/base_time)**2 + (std_devs/mean_times)**2)

# Ideal Linear Scaling
ideal_speedup = threads

plt.figure(figsize=(10, 6))

# Plot Ideal
plt.plot(threads, ideal_speedup, 'k--', label='Ideal Linear Scaling', linewidth=1.5)

# Plot Observed with Error Bars
plt.errorbar(threads, speedup, yerr=speedup_err, fmt='b-o', 
             label='Observed Speedup (Apple M3)', linewidth=2.5, markersize=8, capsize=5)

# Formatting
plt.title('OpenMP Strong Scaling (Apple M3)', fontsize=14, pad=20)
plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Speedup (relative to 1 thread)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xticks(threads)
plt.yticks(np.arange(0, 9, 1))

# Annotations
plt.annotate(f'7.2x Speedup\n(8 cores)', xy=(8, 7.19), xytext=(5, 6),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

# Save
plt.tight_layout()
plt.savefig('docs/openmp_scaling_clean.png', dpi=300)
print("Chart generated: docs/openmp_scaling_clean.png")
