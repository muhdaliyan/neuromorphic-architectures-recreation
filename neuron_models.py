"""
Module 1: Biological Neuron Models

Implements:
  1. Hodgkin-Huxley (HH) model for biophysically realistic action potentials.
  2. Leaky Integrate-and-Fire (LIF) model for hardware-friendly neuron emulation.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

class HodgkinHuxley:
    """
    Hodgkin-Huxley neuron model (1952) simulating action potential dynamics
    via voltage-gated sodium (Na+) and potassium (K+) channels.
    """
    
    # Membrane capacitance (µF/cm²)
    C_m = 1.0
    
    # Maximum conductances (mS/cm²)
    g_Na = 120.0
    g_K  = 36.0
    g_L  = 0.3
    
    # Reversal potentials (mV)
    E_Na = 50.0
    E_K  = -77.0
    E_L  = -54.387
    
    def __init__(self):
        pass
    
    # --- Rate equations (opening/closing rates) ---
    
    def alpha_m(self, V):
        """Na+ activation gate opening rate."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        """Na+ activation gate closing rate."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        """Na+ inactivation gate opening rate."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        """Na+ inactivation gate closing rate."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        """K+ activation gate opening rate."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        """K+ activation gate closing rate."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def I_Na(self, V, m, h):
        """Sodium current current: I_Na = g_Na * m^3 * h * (V - E_Na)"""
        return self.g_Na * m**3 * h * (V - self.E_Na)
    
    def I_K(self, V, n):
        """Potassium current: I_K = g_K * n^4 * (V - E_K)"""
        return self.g_K * n**4 * (V - self.E_K)
    
    def I_L(self, V):
        """Leak current: I_L = g_L * (V - E_L)"""
        return self.g_L * (V - self.E_L)
    
    def derivatives(self, state, t, I_ext):
        """Calculates system derivatives: dV/dt, dm/dt, dh/dt, dn/dt."""
        V, m, h, n = state
        
        dVdt = (I_ext - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        
        return [dVdt, dmdt, dhdt, dndt]
    
    def simulate(self, T=50.0, dt=0.01, I_ext=10.0):
        """Simulates HH neuron model using SciPy's odeint solver."""
        t = np.arange(0.0, T, dt)
        
        # Resting state initial values
        V0 = -65.0
        m0 = 0.05
        h0 = 0.6
        n0 = 0.32
        
        solution = odeint(self.derivatives, [V0, m0, h0, n0], t, args=(I_ext,))
        return t, solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]


class LeakyIntegrateAndFire:
    """
    Leaky Integrate-and-Fire (LIF) neuron model.
    Integrates incoming current over membrane capacitance and resets when hitting threshold.
    """
    
    def __init__(self, tau_m=10.0, V_rest=-65.0, V_thresh=-50.0, 
                 V_reset=-65.0, R_m=10.0):
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.R_m = R_m
    
    def simulate(self, T=100.0, dt=0.1, I_ext_func=None):
        """Simulates LIF neuron using Euler integration."""
        if I_ext_func is None:
            I_ext_func = lambda t: 2.0
        
        t = np.arange(0.0, T, dt)
        V = np.zeros(len(t))
        V[0] = self.V_rest
        spike_times = []
        
        for i in range(1, len(t)):
            dVdt = (-(V[i-1] - self.V_rest) + self.R_m * I_ext_func(t[i-1])) / self.tau_m
            V[i] = V[i-1] + dVdt * dt
            
            if V[i] >= self.V_thresh:
                spike_times.append(t[i])
                V[i] = self.V_reset
                
        return t, V, spike_times


def plot_neuron_models(save_path=None):
    """Generates figures comparing Hodgkin-Huxley and LIF models."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 150,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Biological Neuron Models\n(Nawrocki et al., IEEE TED 2016 — Section II-A)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    color_V  = '#E63946'
    color_m  = '#457B9D'
    color_h  = '#2A9D8F'
    color_n  = '#E9C46A'
    color_LIF = '#264653'
    
    # (a) Hodgkin-Huxley Action Potential
    hh = HodgkinHuxley()
    t_hh, V_hh, m_hh, h_hh, n_hh = hh.simulate(T=50.0, dt=0.01, I_ext=10.0)
    
    ax = axes[0, 0]
    ax.plot(t_hh, V_hh, color=color_V, linewidth=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Voltage (mV)')
    ax.set_title('(a) Hodgkin-Huxley Action Potential', fontweight='bold')
    ax.axhline(y=-55, color='gray', linestyle='--', alpha=0.5, label='Threshold ≈ -55 mV')
    ax.legend(fontsize=9)
    
    # Label the peak
    spike_idx = np.argmax(V_hh[:2000])
    ax.annotate('Depolarization', xy=(t_hh[spike_idx]-2, V_hh[spike_idx]),
                 xytext=(t_hh[spike_idx]-10, V_hh[spike_idx]-20),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=8, color='gray')
    
    # (b) Hodgkin-Huxley Gating Variables
    ax = axes[0, 1]
    ax.plot(t_hh, m_hh, color=color_m, linewidth=2, label='m (Na⁺ activation)')
    ax.plot(t_hh, h_hh, color=color_h, linewidth=2, label='h (Na⁺ inactivation)')
    ax.plot(t_hh, n_hh, color=color_n, linewidth=2, label='n (K⁺ activation)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Gate Probability')
    ax.set_title('(b) HH Gating Variables', fontweight='bold')
    ax.legend(fontsize=9, loc='right')
    ax.set_ylim(-0.05, 1.05)
    
    # (c) LIF Spike Train (Constant Input)
    lif = LeakyIntegrateAndFire(tau_m=10.0, V_rest=-65.0, V_thresh=-50.0, 
                                 V_reset=-65.0, R_m=10.0)
    t_lif, V_lif, spikes_lif = lif.simulate(T=100.0, dt=0.05, 
                                              I_ext_func=lambda t: 2.0)
    
    ax = axes[1, 0]
    ax.plot(t_lif, V_lif, color=color_LIF, linewidth=1.5)
    for spike_t in spikes_lif:
        ax.axvline(x=spike_t, color=color_V, alpha=0.6, linewidth=1, linestyle='-')
    ax.axhline(y=-50, color='gray', linestyle='--', alpha=0.5, label='Threshold = -50 mV')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Voltage (mV)')
    ax.set_title('(c) Leaky Integrate-and-Fire: Constant Input', fontweight='bold')
    ax.legend(fontsize=9)
    
    if len(spikes_lif) > 1:
        firing_rate = len(spikes_lif) / (100.0 / 1000.0)
        ax.text(0.95, 0.15, f'Firing rate: {firing_rate:.0f} Hz\nSpikes: {len(spikes_lif)}',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # (d) LIF: Response to Varying Current
    def step_current(t):
        if t < 25:
            return 1.0
        elif t < 50:
            return 1.8
        elif t < 75:
            return 2.5
        else:
            return 4.0
    
    t_lif2, V_lif2, spikes_lif2 = lif.simulate(T=100.0, dt=0.05, 
                                                  I_ext_func=step_current)
    
    ax = axes[1, 1]
    ax.plot(t_lif2, V_lif2, color=color_LIF, linewidth=1.5)
    for spike_t in spikes_lif2:
        ax.axvline(x=spike_t, color=color_V, alpha=0.6, linewidth=1)
    ax.axhline(y=-50, color='gray', linestyle='--', alpha=0.5)
    
    ax.axvspan(0, 25, alpha=0.08, color='blue', label='I=1.0 nA (sub-threshold)')
    ax.axvspan(25, 50, alpha=0.08, color='green', label='I=1.8 nA (slow)')
    ax.axvspan(50, 75, alpha=0.08, color='orange', label='I=2.5 nA (moderate)')
    ax.axvspan(75, 100, alpha=0.08, color='red', label='I=4.0 nA (fast)')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Voltage (mV)')
    ax.set_title('(d) LIF: Varying Input Current', fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
    
    plt.show()
    return fig

if __name__ == "__main__":
    plot_neuron_models(save_path='figures/fig1_neuron_models.png')
