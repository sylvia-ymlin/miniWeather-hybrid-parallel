#!/bin/bash
# Test if PNetCDF is available and working
# Run a small test to verify output generation

set -e

cd "$(dirname "$0")/.."
BUILD_DIR="${1:-build}"

echo "=== PNetCDF Diagnostic Test ==="
echo ""

# Check if executable exists
if [ ! -f "$BUILD_DIR/miniWeather_mpi" ]; then
    echo "❌ Error: $BUILD_DIR/miniWeather_mpi not found"
    exit 1
fi

# Allow OpenMPI to run as root
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Clean up any existing output
rm -f output_test.nc

echo "Running small test (Thermal, 100x50, 10s, output every 5s)..."
echo "This should generate output_test.nc if PNetCDF is enabled"
echo ""

# Run a very small test
mpirun -n 1 "$BUILD_DIR/miniWeather_mpi" \
    --data 2 \
    --nx 100 \
    --nz 50 \
    --time 10 \
    --freq 5 \
    2>&1 | tail -20

echo ""
echo "=== Checking Results ==="

if [ -f output_test.nc ]; then
    echo "✅ SUCCESS: output_test.nc was generated!"
    echo "   PNetCDF is working correctly."
    ls -lh output_test.nc
    echo ""
    echo "You can now run full visualization generation:"
    echo "  bash scripts/generate_visualizations.sh"
    mv output_test.nc output_thermal_test.nc
elif [ -f output.nc ]; then
    echo "✅ SUCCESS: output.nc was generated!"
    echo "   PNetCDF is working correctly."
    ls -lh output.nc
    echo ""
    echo "Renaming to output_thermal_test.nc..."
    mv output.nc output_thermal_test.nc
else
    echo "❌ FAILED: No output file generated"
    echo ""
    echo "Possible reasons:"
    echo "  1. PNetCDF is not installed on the system"
    echo "  2. Build was not compiled with PNetCDF enabled"
    echo "  3. Check the output above for error messages"
    echo ""
    echo "To enable PNetCDF, you need to:"
    echo "  1. Install PNetCDF: apt-get install libpnetcdf-dev (Ubuntu/Debian)"
    echo "  2. Rebuild: cd build && cmake .. -DENABLE_PNETCDF=ON && make"
    exit 1
fi

