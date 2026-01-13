#!/bin/bash
# Generate NetCDF output files for all scenarios for visualization
# Requires: PNetCDF enabled build

set -e

cd "$(dirname "$0")/.."
BUILD_DIR="${1:-build}"

if [ ! -f "$BUILD_DIR/miniWeather_mpi" ]; then
    echo "Error: $BUILD_DIR/miniWeather_mpi not found"
    echo "Please build with PNetCDF enabled:"
    echo "  cd $BUILD_DIR && cmake .. -DENABLE_PNETCDF=ON && make"
    exit 1
fi

echo "Generating visualization data for all scenarios..."
echo "Note: This requires PNetCDF enabled build"
echo ""

# Colliding Thermals (DATA_SPEC=1, sim_time=700, out_freq=100)
echo "[1/3] Running Colliding Thermals (700s)..."
mpirun -n 1 "$BUILD_DIR/miniWeather_mpi" --data 1 --nx 400 --nz 200 --time 700 --freq 100
mv output.nc output_collision_long.nc 2>/dev/null || echo "  Note: output.nc may not exist if PNetCDF not enabled"

# Mountain Gravity Waves (DATA_SPEC=3, sim_time=1500, out_freq=300)
echo "[2/3] Running Mountain Gravity Waves (1500s)..."
mpirun -n 1 "$BUILD_DIR/miniWeather_mpi" --data 3 --nx 400 --nz 200 --time 1500 --freq 300
mv output.nc output_gravity_long.nc 2>/dev/null || echo "  Note: output.nc may not exist if PNetCDF not enabled"

# Injection (DATA_SPEC=6, sim_time=1200, out_freq=300)
echo "[3/3] Running Injection (1200s)..."
mpirun -n 1 "$BUILD_DIR/miniWeather_mpi" --data 6 --nx 400 --nz 200 --time 1200 --freq 300
mv output.nc output_injection_long.nc 2>/dev/null || echo "  Note: output.nc may not exist if PNetCDF not enabled"

echo ""
echo "Done! Now run: python3 scripts/visualize_pro.py"

