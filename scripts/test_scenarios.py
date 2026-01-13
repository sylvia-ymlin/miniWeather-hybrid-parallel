#!/usr/bin/env python3
"""
Automated Test Suite for miniWeather Hybrid-Parallel
Tests all simulation scenarios and validates mass conservation.

Usage:
    python3 test_scenarios.py --exe ./build/miniWeather_serial
    python3 test_scenarios.py --exe ./build/miniWeather_mpi --mpi 4
"""

import argparse
import subprocess
import sys
import os
import re

# Test scenario configurations matching original miniWeather
# Format: (name, data_spec_int, sim_time, nx, nz, expected_mass_change)
# Note: injection scenario intentionally adds mass, so we set a higher threshold
TEST_SCENARIOS = [
    ("thermal",         2, 100, 200, 100, 1e-13),  # Rising thermal
    ("collision",       1, 100, 200, 100, 1e-13),  # Colliding thermals
    ("gravity_waves",   3, 100, 200, 100, 1e-13),  # Mountain gravity waves
    ("density_current", 5, 100, 200, 100, 1e-13),  # Density current
    ("injection",       6, 100, 200, 100, 0.1),    # Injection (adds mass by design!)
]

# Quick test configurations (faster, for CI)
QUICK_TEST_SCENARIOS = [
    ("thermal",         2, 10, 100, 50, 1e-13),
    ("density_current", 5, 10, 100, 50, 1e-13),
]

def parse_output(output):
    """Parse simulation output for mass and energy changes."""
    mass_change = None
    energy_change = None
    
    for line in output.split('\n'):
        if 'd_mass:' in line:
            try:
                mass_change = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        if 'd_te:' in line:
            try:
                energy_change = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
    
    return mass_change, energy_change

def run_test(exe, scenario, mpi_ranks=None, verbose=False):
    """Run a single test scenario."""
    name, data_spec, sim_time, nx, nz, max_mass_change = scenario
    
    # Build command
    cmd = []
    env = os.environ.copy()
    if mpi_ranks and mpi_ranks > 1:
        cmd = ['mpirun', '-np', str(mpi_ranks)]
        # Allow OpenMPI to run as root (environment variables are harmless for MPICH)
        # Check if running as root
        try:
            if os.geteuid() == 0:  # root user (Unix/Linux only)
                env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
                env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
        except AttributeError:
            pass  # Windows or geteuid() not available
    cmd.append(exe)
    cmd.extend(['--nx', str(nx), '--nz', str(nz), 
                '--time', str(sim_time), '--data', str(data_spec),
                '--freq', '-1'])  # Disable output for faster testing
    
    if verbose:
        print(f"  Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            return False, f"Non-zero exit code: {result.returncode}\n{output}"
        
        mass_change, energy_change = parse_output(output)
        
        if mass_change is None:
            return False, f"Could not parse mass change from output:\n{output}"
        
        # Check mass conservation (should be at machine precision for most cases)
        # Note: injection scenario intentionally adds mass, has relaxed threshold
        if abs(mass_change) > max_mass_change:
            return False, f"Mass conservation failed: {mass_change} > {max_mass_change}"
        
        te_str = f"{energy_change:.2e}" if energy_change is not None else "N/A"
        return True, f"d_mass={mass_change:.2e}, d_te={te_str}"
        
    except subprocess.TimeoutExpired:
        return False, "Test timed out (300s)"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='miniWeather Test Suite')
    parser.add_argument('--exe', required=True, help='Path to executable')
    parser.add_argument('--mpi', type=int, default=0, help='Number of MPI ranks (0 for serial)')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--scenario', type=str, help='Run specific scenario only')
    args = parser.parse_args()
    
    if not os.path.exists(args.exe):
        print(f"ERROR: Executable not found: {args.exe}")
        sys.exit(1)
    
    scenarios = QUICK_TEST_SCENARIOS if args.quick else TEST_SCENARIOS
    
    if args.scenario:
        # When testing specific scenario, always search in full test list
        scenarios = [s for s in TEST_SCENARIOS if s[0] == args.scenario]
        if not scenarios:
            print(f"ERROR: Unknown scenario: {args.scenario}")
            print(f"Available: {[s[0] for s in TEST_SCENARIOS]}")
            sys.exit(1)
        # For individual tests, use quick parameters (shorter sim_time)
        if args.quick:
            name, data_spec, _, _, _, max_mass = scenarios[0]
            scenarios = [(name, data_spec, 10, 100, 50, max_mass)]
    
    print("=" * 60)
    print("miniWeather Automated Test Suite")
    print("=" * 60)
    print(f"Executable: {args.exe}")
    print(f"MPI Ranks:  {args.mpi if args.mpi else 'Serial'}")
    print(f"Test Mode:  {'Quick' if args.quick else 'Full'}")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for scenario in scenarios:
        name = scenario[0]
        print(f"\n[TEST] {name.upper()}")
        print("-" * 40)
        
        success, message = run_test(args.exe, scenario, args.mpi, args.verbose)
        
        if success:
            print(f"  ✓ PASSED: {message}")
            passed += 1
        else:
            print(f"  ✗ FAILED: {message}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()

