cmake_minimum_required(VERSION 3.10)
project(miniWeather CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP)

# Default Parameters
if(NOT DEFINED NX)
    set(NX 100)
endif()
if(NOT DEFINED NZ)
    set(NZ 50)
endif()
if(NOT DEFINED SIM_TIME)
    set(SIM_TIME 1000)
endif()
if(NOT DEFINED OUT_FREQ)
    set(OUT_FREQ 10)
endif()
if(NOT DEFINED DATA_SPEC)
    set(DATA_SPEC 2) # DATA_SPEC_THERMAL = 2
endif()

# Compiler Definitions
add_definitions(-D_NX=${NX})
add_definitions(-D_NZ=${NZ})
add_definitions(-D_SIM_TIME=${SIM_TIME})
add_definitions(-D_OUT_FREQ=${OUT_FREQ})
add_definitions(-D_DATA_SPEC=${DATA_SPEC})

# ============================================================================
# Target 1: Serial Baseline (with optional OpenMP for multi-threading)
# ============================================================================
add_executable(miniWeather_serial src/miniWeather_serial.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(miniWeather_serial PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Target 2: MPI + OpenMP (Hybrid CPU Parallelism)
# ============================================================================
add_executable(miniWeather_mpi src/miniWeather_mpi.cpp)
target_link_libraries(miniWeather_mpi PUBLIC MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(miniWeather_mpi PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# PNetCDF support (Parallel NetCDF)
# ============================================================================
option(ENABLE_PNETCDF "Enable Parallel NetCDF output" OFF)
if(ENABLE_PNETCDF)
    # Try to find PNetCDF using pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PNETCDF IMPORTED_TARGET pnetcdf)
    endif()
    if(PNETCDF_FOUND)
        message(STATUS "Found PNetCDF: ${PNETCDF_LIBRARIES}")
        # Link PNetCDF to MPI target
        target_link_libraries(miniWeather_mpi PUBLIC PkgConfig::PNETCDF)
    else()
        message(WARNING "PNetCDF not found, output will be disabled")
        add_definitions(-D_NO_PNETCDF)
    endif()
else()
    add_definitions(-D_NO_PNETCDF)
endif()

# ============================================================================
# Target 3: MPI + OpenACC (GPU Offloading)
# Note: Manually compile with nvc++ due to complex linker dependencies
# See PROJECT_ANALYSIS.md for instructions.
# ============================================================================

# ============================================================================
# Target 4: MPI + OpenMP 4.5 Target (GPU Offloading via OpenMP)
# Requires Clang/LLVM with offloading support or NVIDIA HPC SDK
# Build: cmake -DCMAKE_CXX_COMPILER=clang++ -DENABLE_OMP_TARGET=ON ..
# ============================================================================
option(ENABLE_OMP_TARGET "Build OpenMP Target GPU version" OFF)
if(ENABLE_OMP_TARGET)
    add_executable(miniWeather_omp_target src/miniWeather_mpi_openmp45.cpp)
    target_link_libraries(miniWeather_omp_target PUBLIC MPI::MPI_CXX)
    # Adjust flags based on your compiler (example for nvc++)
    target_compile_options(miniWeather_omp_target PRIVATE -mp=gpu -Minfo=mp)
    message(STATUS "OpenMP Target offloading enabled. Requires compatible compiler.")
endif()

# ============================================================================
# Testing - Comprehensive Test Suite
# ============================================================================
enable_testing()

# Quick sanity test for serial version
add_test(NAME Serial_Quick
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_serial> --quick
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Quick sanity test for MPI version  
add_test(NAME MPI_Quick
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_mpi> --mpi 2 --quick
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Full test suite for serial (all 5 scenarios)
add_test(NAME Serial_Full
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_serial>
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Full test suite for MPI (all 5 scenarios, 4 ranks)
add_test(NAME MPI_Full
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_mpi> --mpi 4
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Individual scenario tests (for debugging specific issues)
add_test(NAME Thermal_Serial
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_serial> --scenario thermal --quick
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME Collision_Serial
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_serial> --scenario collision --quick
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME GravityWaves_Serial
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_serial> --scenario gravity_waves --quick
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME DensityCurrent_Serial
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_serial> --scenario density_current --quick
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME Injection_Serial
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_scenarios.py
                 --exe $<TARGET_FILE:miniWeather_serial> --scenario injection --quick
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Set test labels for filtering
set_tests_properties(Serial_Quick MPI_Quick PROPERTIES LABELS "quick")
set_tests_properties(Serial_Full MPI_Full PROPERTIES LABELS "full")
set_tests_properties(Thermal_Serial Collision_Serial GravityWaves_Serial 
                     DensityCurrent_Serial Injection_Serial PROPERTIES LABELS "individual")

