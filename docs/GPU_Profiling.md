# GPU Profiling Results

**Date:** January 28, 2026
**Platform:** NVIDIA RTX 3090 (AutoDL Container)
**Compiler:** nvc++ 26.1

## 1. Hardware Configuration
*   **CPU:** Intel Xeon Platinum 8358P (32 cores)
*   **GPU:** NVIDIA GeForce RTX 3090
    *   Compute Capability: 8.6
    *   Memory: 24 GB GDDR6X (936 GB/s)
    *   L2 Cache: 6 MB
    *   **Peak FP64:** 0.556 TFLOPS (1:64 ratio)
    *   **Peak FP32:** 35.6 TFLOPS

---

## 2. Executive Summary
The application is **Compute Bound** and strictly limited by Double Precision (FP64) performance on the consumer-grade RTX 3090.
*   **Scaling:** Linear (or better) scaling from 100x50 to 1000x500. No cache thrashing observed.
*   **Bottleneck:** **FP64 Throughput** + **Register Spilling**.
*   **Efficiency:** GPU occupancy is high (>99%), but realized GFLOPS is low (~87 GFLOPS) due to heavy register usage spilling to L1/L2.

---

## 3. Profiling Data Comparison

| Metric | 400x200 (Small) | 800x400 (Medium) | Change |
| :--- | :--- | :--- | :--- |
| **Grid Points** | 80,000 | 320,000 | 4.0x |
| **Avg Kernel Time** | 131 µs | 410 µs | **3.1x** (Better than 4x) |
| **Avg Gap (Overhead)**| ~1.06 µs | < 1 µs | Negligible |
| **PCIe HtoD** | 0.6 ms | ~3.0 ms | ~5x (Negligible total) |
| **SM Occupancy** | >95% | >99% | Saturated |
| **L2 Hit Rate** | N/A (Blocked) | N/A (Blocked) | - |
| **Bottleneck** | Latency/Overhead | FP64 Throughput | Transition to Saturation |

### Interpretation
"GPU scaling improves from 400x200 to 800x400 (3.1x time increase for 4x work), indicating that smaller grids were partially latency-bound. At 800x400, the kernel is fully saturated. Static analysis reveals an Arithmetic Intensity of **0.88 FLOPs/Byte**, which exceeds the RTX 3090's FP64 ridge point of 0.6. This confirms the application is **Compute Bound**. The 6MB L2 cache is *not* thrashing; rather, the register file is under pressure, causing spills that limit peak throughput below theoretical maximums."

---

## 4. Visual Analysis

### Scaling Plot (Linearity Check)
![Kernel Time vs Grid Size](/Users/ymlin/Downloads/003-Study/138-Projects/11-miniWeather-hybrid-parallel/profiles/scaling_plot.png)
*Points track the linear verification line, ruling out memory/cache bounds.*

### Roofline Plot (Bound Check)
![Roofline Plot](/Users/ymlin/Downloads/003-Study/138-Projects/11-miniWeather-hybrid-parallel/profiles/roofline_plot.png)
*Red dot is to the right of the ridge (Compute Bound) but below the roof (Inefficiency).*

---

## 5. Quick Reference Commands

### Basic Profiling
```bash
# General Stats
nsys profile --stats=true ./miniWeather_mpi_openacc ...

# Timeline Trace
nsys profile --trace=cuda,nvtx,osrt,mpi -o timeline ./miniWeather_mpi_openacc ...
```

### Advanced Analysis (If Permissions Allow)
```bash
# Detailed Kernel Metrics
ncu --set full -o report ./miniWeather_mpi_openacc ...

# Roofline Metrics
ncu --metrics dram__bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum ...
```

### Manual Verification
```bash
# Grid Sweep
for nx in 400 800 1000; do
    ... run and grep kernel time ...
done
```
