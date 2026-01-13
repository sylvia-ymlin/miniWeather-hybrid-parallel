#!/bin/bash
# Generate NetCDF output files for all scenarios for visualization
# Requires: PNetCDF enabled build

set -e

cd "$(dirname "$0")/.."
BUILD_DIR="${1:-build}"
MPI_RANKS="${2:-4}"  # Default to 4 processes, can be overridden

if [ ! -f "$BUILD_DIR/miniWeather_mpi" ]; then
    echo "Error: $BUILD_DIR/miniWeather_mpi not found"
    echo "Please build with PNetCDF enabled:"
    echo "  cd $BUILD_DIR && cmake .. -DENABLE_PNETCDF=ON && make"
    exit 1
fi

# Allow OpenMPI to run as root (for container environments)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

echo "Generating visualization data for all scenarios..."
echo "Note: This requires PNetCDF enabled build"
echo "Using $MPI_RANKS MPI processes for faster execution"
echo ""

# Colliding Thermals (DATA_SPEC=1, sim_time=200, out_freq=50) - Reduced for faster generation
echo "[1/3] Running Colliding Thermals (200s) with $MPI_RANKS processes..."
mpirun -n $MPI_RANKS "$BUILD_DIR/miniWeather_mpi" --data 1 --nx 200 --nz 100 --time 200 --freq 50
if [ -f output.nc ]; then
    mv output.nc output_collision_long.nc
    echo "  ✓ Generated output_collision_long.nc"
else
    echo "  ✗ ERROR: output.nc not generated (PNetCDF may not be enabled/installed)"
    echo "  Please build with: cmake .. -DENABLE_PNETCDF=ON && make"
fi

# Mountain Gravity Waves (DATA_SPEC=3, sim_time=400, out_freq=100) - Reduced for faster generation
echo "[2/3] Running Mountain Gravity Waves (400s) with $MPI_RANKS processes..."
mpirun -n $MPI_RANKS "$BUILD_DIR/miniWeather_mpi" --data 3 --nx 200 --nz 100 --time 400 --freq 100
if [ -f output.nc ]; then
    mv output.nc output_gravity_long.nc
    echo "  ✓ Generated output_gravity_long.nc"
else
    echo "  ✗ ERROR: output.nc not generated (PNetCDF may not be enabled/installed)"
fi

# Injection (DATA_SPEC=6, sim_time=300, out_freq=75) - Reduced for faster generation
echo "[3/3] Running Injection (300s) with $MPI_RANKS processes..."
mpirun -n $MPI_RANKS "$BUILD_DIR/miniWeather_mpi" --data 6 --nx 200 --nz 100 --time 300 --freq 75
if [ -f output.nc ]; then
    mv output.nc output_injection_long.nc
    echo "  ✓ Generated output_injection_long.nc"
else
    echo "  ✗ ERROR: output.nc not generated (PNetCDF may not be enabled/installed)"
fi

echo ""
echo "Done! Now run: python3 scripts/visualize_pro.py"

