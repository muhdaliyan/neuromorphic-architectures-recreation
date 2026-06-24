"""
Neuromorphic Architectures - Master Simulation Runner

Runs all five simulation modules to reproduce the results from Nawrocki et al. (IEEE TED 2016).
Saves generated figures to the 'figures/' directory.
"""

import os
import sys
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments

def main():
    print("=" * 70)
    print("  NEUROMORPHIC ARCHITECTURES: COMPUTATIONAL RECREATION")
    print("  Based on: Nawrocki et al., IEEE TED, Vol. 63, No. 10, 2016")
    print("=" * 70)
    print()
    
    os.makedirs('figures', exist_ok=True)
    results = {}
    start_total = time.time()
    
    # --- Module 1: Neuron Models ---
    print("-" * 60)
    print("Running Module 1: Biological Neuron Models (HH & LIF)")
    print("-" * 60)
    try:
        start = time.time()
        from neuron_models import HodgkinHuxley, LeakyIntegrateAndFire, plot_neuron_models
        
        hh = HodgkinHuxley()
        t, V, m, h, n = hh.simulate(T=50, dt=0.01, I_ext=10)
        print(f"  HH Model: Peak voltage = {max(V):.1f} mV")
        
        lif = LeakyIntegrateAndFire()
        t, V, spikes = lif.simulate(T=100, dt=0.1)
        print(f"  LIF Model: {len(spikes)} spikes generated in 100 ms")
        
        plot_neuron_models(save_path='figures/fig1_neuron_models.png')
        
        elapsed = time.time() - start
        results['Module 1'] = f'Complete ({elapsed:.1f}s)'
        print(f"  Done in {elapsed:.1f}s\n")
    except Exception as e:
        results['Module 1'] = f'Error: {e}'
        print(f"  Error: {e}\n")
    
    # --- Module 2: Synaptic Models ---
    print("-" * 60)
    print("Running Module 2: Synaptic Dynamics and Plasticity (Current Mirror & STDP)")
    print("-" * 60)
    try:
        start = time.time()
        from synaptic_models import CurrentMirrorSynapse, STDP, plot_synaptic_models
        
        synapse = CurrentMirrorSynapse()
        t, Q, I_EX = synapse.simulate([10, 20, 30, 40, 50], T=80)
        print(f"  Synapse: Peak current = {max(I_EX):.3f}")
        
        stdp = STDP()
        dw_ltp = stdp.weight_change(10)
        dw_ltd = stdp.weight_change(-10)
        print(f"  STDP: LTP(dt=+10ms) = +{dw_ltp:.4f}, LTD(dt=-10ms) = {dw_ltd:.4f}")
        
        plot_synaptic_models(save_path='figures/fig2_synaptic_models.png')
        
        elapsed = time.time() - start
        results['Module 2'] = f'Complete ({elapsed:.1f}s)'
        print(f"  Done in {elapsed:.1f}s\n")
    except Exception as e:
        results['Module 2'] = f'Error: {e}'
        print(f"  Error: {e}\n")
    
    # --- Module 3: Memristive Devices ---
    print("-" * 60)
    print("Running Module 3: Memristive Device Simulation (Digital, Analog, STP/STD)")
    print("-" * 60)
    try:
        start = time.time()
        from memristive_devices import DigitalMemristor, AnalogMemristor, plot_memristive_devices
        
        digital = DigitalMemristor()
        V, I = digital.iv_curve()
        print(f"  Digital Memristor: R_off/R_on ratio = {digital.R_off/digital.R_on:.0f}x")
        
        analog = AnalogMemristor()
        V, I = analog.iv_curve()
        print(f"  Analog Memristor: R range = {analog.R_min:.0f} to {analog.R_max:.0f} Ohms")
        
        plot_memristive_devices(save_path='figures/fig3_memristive_devices.png')
        
        elapsed = time.time() - start
        results['Module 3'] = f'Complete ({elapsed:.1f}s)'
        print(f"  Done in {elapsed:.1f}s\n")
    except Exception as e:
        results['Module 3'] = f'Error: {e}'
        print(f"  Error: {e}\n")
    
    # --- Module 4: Crossbar Network ---
    print("-" * 60)
    print("Running Module 4: Memristive Crossbar Network (3x3 Classification)")
    print("-" * 60)
    try:
        start = time.time()
        from crossbar_network import MemristiveCrossbar, PATTERNS, plot_crossbar_network
        
        plot_crossbar_network(save_path='figures/fig4_crossbar_network.png')
        
        elapsed = time.time() - start
        results['Module 4'] = f'Complete ({elapsed:.1f}s)'
        print(f"  Done in {elapsed:.1f}s\n")
    except Exception as e:
        results['Module 4'] = f'Error: {e}'
        print(f"  Error: {e}\n")
    
    # --- Module 5: Systems Comparison ---
    print("-" * 60)
    print("Running Module 5: Neuromorphic Systems Comparison")
    print("-" * 60)
    try:
        start = time.time()
        from systems_comparison import print_table, plot_systems_comparison
        
        print_table()
        plot_systems_comparison(save_path='figures/fig5_systems_comparison.png')
        
        elapsed = time.time() - start
        results['Module 5'] = f'Complete ({elapsed:.1f}s)'
        print(f"  Done in {elapsed:.1f}s\n")
    except Exception as e:
        results['Module 5'] = f'Error: {e}'
        print(f"  Error: {e}\n")
    
    # --- Execution Summary ---
    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print("                               SUMMARY")
    print("=" * 70)
    for module, status in results.items():
        print(f"  {module}: {status:<50}")
    print("=" * 70)
    print(f"  Total Execution Time: {total_time:.1f}s")
    print("  Output figures saved in figures/")
    print("=" * 70)

if __name__ == "__main__":
    main()
