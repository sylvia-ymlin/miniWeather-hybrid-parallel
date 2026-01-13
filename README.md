# miniWeather: High-Performance Hybrid Parallel Atmospheric Solver

## Overview

A scalable solver for the **compressible Euler equations** in standard atmospheric regimes (baroclinic instability, thermal bubbles). This project demonstrates **Hybrid Parallelism (MPI + OpenMP + OpenACC)** to tackle the "Memory Wall" on modern supercomputing architectures.

---

## Simulation Output

### Rising Thermal (500s)
![Thermal 500s](docs/thermal_500s.png)
_Classic "mushroom cloud" convective plumes with counter-rotating vortices at 500 seconds._

### Rising Thermal (1000s)
![Thermal 1000s](docs/thermal_999s.png)
_Fully developed turbulent convection with chaotic vortex structures at 1000 seconds._

### Density Current / Cold Front (600s)
![Density 600s](docs/density_600s.png)
_Cold air mass propagating along the surface, characteristic of atmospheric fronts._

---

## Mathematical Foundation

### Governing Equations (2D Compressible Euler)

The solver integrates the conservation laws for density, momentum, and potential temperature:

$$
\frac{\partial}{\partial t}
\begin{pmatrix} \rho \\ \rho u \\ \rho w \\ \rho \theta \end{pmatrix}
+ \frac{\partial}{\partial x}
\begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u w \\ \rho u \theta \end{pmatrix}
+ \frac{\partial}{\partial z}
\begin{pmatrix} \rho w \\ \rho u w \\ \rho w^2 + p \\ \rho w \theta \end{pmatrix}
=
\begin{pmatrix} 0 \\ 0 \\ -\rho g \\ 0 \end{pmatrix}
$$

where:
- $\rho$ : density (kg/m³)
- $u, w$ : horizontal and vertical velocity (m/s)
- $\theta$ : potential temperature (K)
- $p = C_0 (\rho \theta)^\gamma$ : pressure via equation of state
- $g = 9.8$ m/s² : gravitational acceleration

### Boundary Conditions

| Boundary | Type | Implementation |
|:---------|:-----|:---------------|
| **Horizontal (x)** | Periodic | `state[i-hs] = state[nx+i]` |
| **Vertical Top** | Rigid Wall | $w = 0$, zero-gradient for $\rho, \theta$ |
| **Vertical Bottom** | Rigid Wall | $w = 0$, hydrostatic extrapolation |

---

## Performance Highlights

| Metric | Result | Platform |
|:-------|:-------|:---------|
| **OpenMP Scaling** | **9.5× speedup** (12 threads, 79% efficiency) | Intel Xeon Platinum 8358P |
| **Hybrid Efficiency** | **+2.1% faster** than pure OpenMP (MPI 2×4) | 15 vCPU Cloud Instance |
| **GPU Acceleration** | Verified on RTX 3090 | NVIDIA OpenACC/OpenMP Target |
| **Mass Conservation** | $|\Delta m| < 10^{-13}$ | Machine precision |

---

## Technical Architecture

| Component | Implementation Details |
|:----------|:-----------------------|
| **Domain Decomposition** | 1D Cartesian (x-direction), Non-Blocking MPI (`MPI_Isend`/`MPI_Irecv`) |
| **Numerical Core** | 4th-Order Finite Volume + 3rd-Order TVD Runge-Kutta |
| **Parallel Strategy** | Hybrid: MPI (inter-node) + OpenMP (intra-node) + OpenACC (GPU) |
| **Halo Exchange** | 4-cell ghost zone, asynchronous communication |

### Domain Decomposition & Halo Exchange

```
┌─────────────────────────────────────────────────────────────┐
│                    Global Domain (nx × nz)                  │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   Rank 0     │   Rank 1     │   Rank 2     │   Rank 3      │
│  ┌────┬────┐ │ ┌────┬────┐  │ ┌────┬────┐  │ ┌────┬────┐   │
│  │Halo│Data│◄─►│Halo│Data│◄──►│Halo│Data│◄──►│Halo│Data│   │
│  └────┴────┘ │ └────┴────┘  │ └────┴────┘  │ └────┴────┘   │
│   hs=4 cells │              │              │               │
└──────────────┴──────────────┴──────────────┴───────────────┘
                    ◄─── MPI_Isend/Irecv ───►
```

---

## Quick Start

### Prerequisites

| Dependency | Version | Required For |
|:-----------|:--------|:-------------|
| CMake | ≥ 3.10 | Build system |
| GCC/Clang | ≥ 7.0 | C++11 support |
| OpenMPI/MPICH | any | MPI parallelism |
| OpenMP | ≥ 4.5 | Thread parallelism |
| NVIDIA HPC SDK | ≥ 21.0 | OpenACC GPU offloading |
| PNetCDF | (optional) | Parallel I/O |

### Build Options

```bash
# 1. Standard Build (MPI + OpenMP)
mkdir build && cd build
cmake .. 
make -j

# 2. With GPU Offloading (OpenACC - requires nvc++)
nvc++ -acc -Minfo=accel -o miniWeather_openacc src/miniWeather_mpi_openacc.cpp -lmpi

# 3. With OpenMP Target Offloading (requires clang++ or nvc++)
cmake .. -DENABLE_OMP_TARGET=ON -DCMAKE_CXX_COMPILER=nvc++
make -j

# 4. Custom Grid Size
cmake .. -DNX=400 -DNZ=200 -DSIM_TIME=100
```

### Running Simulations

```bash
# Serial with OpenMP threading (4 threads)
OMP_NUM_THREADS=4 ./miniWeather_serial --nx 400 --nz 200 --time 10

# MPI only (4 Ranks)
mpirun -n 4 ./miniWeather_mpi --nx 400 --nz 200 --time 10

# Hybrid MPI + OpenMP (2 Ranks × 4 Threads = 8 cores)
OMP_NUM_THREADS=4 mpirun -n 2 ./miniWeather_mpi --nx 400 --nz 200 --time 10
```

---

## Scaling Results

### OpenMP Thread Scaling & Hybrid Comparison

![OpenMP Scaling](docs/openmp_scaling_results.png)

_Intel Xeon Platinum 8358P @ 2.60GHz, 400×200 grid, 10s simulation. The hybrid 2×4 configuration outperforms pure OpenMP by reducing memory contention._

### CPU vs GPU (OpenACC) Performance

![CPU GPU Comparison](docs/cpu_gpu_comparison.png)

_**Measured data**: GPU achieves **15.6× speedup** at 400×200 grid. All tests on AutoDL Cloud (RTX 3090 + Intel Xeon Platinum)._

### MPI Strong/Weak Scaling Analysis

![MPI Scaling](docs/scaling_results.png)

### Performance Analysis & Known Limitations

#### Why Hybrid (1×4) is Slower than Pure MPI (4×1)?

At small core counts (≤4), the **OpenMP thread creation overhead** dominates:
- Thread fork/join cost: ~10-50 μs per parallel region
- With 1000+ timesteps, this accumulates to significant overhead
- Pure MPI avoids this by using persistent processes

**Recommendation**: Hybrid mode benefits emerge at **≥8 cores** where reduced MPI communication outweighs thread overhead.

#### Weak Scaling Efficiency Drop (33% at 4 Ranks)

The observed efficiency degradation is due to **memory bandwidth saturation**:
- Apple M-series unified memory: ~200 GB/s shared bandwidth
- 4 MPI processes compete for the same memory controller
- **Planned optimization**: Implement communication-computation overlap using `MPI_Ialltoall` with pipelined halo updates

---

## File Structure

```
miniWeather-hybrid-parallel/
├── src/
│   ├── miniWeather_serial.cpp      # OpenMP-threaded baseline
│   ├── miniWeather_mpi.cpp         # MPI + OpenMP hybrid
│   ├── miniWeather_mpi_openacc.cpp # MPI + OpenACC GPU
│   └── miniWeather_mpi_openmp45.cpp# MPI + OpenMP Target GPU
├── scripts/
│   ├── validate.py                 # Physics validation
│   ├── plot_openmp_scaling.py      # Generate scaling charts
│   └── scaling_study.py            # Automated benchmarking
├── docs/
│   ├── project_analysis.md         # Detailed performance analysis
│   └── *.png                       # Visualization outputs
└── CMakeLists.txt
```

---

## License

BSD 3-Clause License. Based on [ORNL miniWeather](https://github.com/mrnorman/miniWeather).
