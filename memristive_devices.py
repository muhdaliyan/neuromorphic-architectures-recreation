"""
Module 3: Memristive Device Simulation

Implements:
  1. Digital (two-state) memristor I-V curve hysteresis.
  2. Analog (multi-level) memristor I-V curve hysteresis.
  3. Short-Term Potentiation (STP) dynamics.
  4. Short-Term Depression (STD) dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class DigitalMemristor:
    """
    Two-state digital/bistable memristive device model.
    Switches between High Resistance State (HRS) and Low Resistance State (LRS)
    at specific voltage thresholds (V_set and V_reset).
    """
    
    def __init__(self, R_on=1e3, R_off=1e5, V_set=0.8, V_reset=-0.6):
        self.R_on = R_on
        self.R_off = R_off
        self.V_set = V_set
        self.V_reset = V_reset
    
    def iv_curve(self, V_max=1.5, n_points=1000):
        """Generates hysteretic I-V curve via a voltage sweep loop."""
        V_up = np.linspace(0, V_max, n_points)
        V_down_pos = np.linspace(V_max, 0, n_points)
        V_down_neg = np.linspace(0, -V_max, n_points)
        V_up_neg = np.linspace(-V_max, 0, n_points)
        
        V = np.concatenate([V_up, V_down_pos, V_down_neg, V_up_neg])
        I = np.zeros_like(V)
        state = 'HRS'
        
        for i, v in enumerate(V):
            if state == 'HRS' and v > self.V_set:
                state = 'LRS'
            elif state == 'LRS' and v < self.V_reset:
                state = 'HRS'
            
            R = self.R_on if state == 'LRS' else self.R_off
            I[i] = v / R
            
        return V, I


class AnalogMemristor:
    """
    Multi-level analog memristive device model.
    Conductance changes continuously based on the state variable x.
    """
    
    def __init__(self, R_min=1e3, R_max=1e5, eta=1.0):
        self.R_min = R_min
        self.R_max = R_max
        self.eta = eta
    
    def iv_curve(self, V_max=1.2, n_points=1000):
        """Generates multi-level analog hysteretic I-V curve."""
        V_up = np.linspace(0, V_max, n_points)
        V_down_pos = np.linspace(V_max, 0, n_points)
        V_down_neg = np.linspace(0, -V_max, n_points)
        V_up_neg = np.linspace(-V_max, 0, n_points)
        
        V = np.concatenate([V_up, V_down_pos, V_down_neg, V_up_neg])
        I = np.zeros_like(V)
        
        x = 0.1  # State variable (0: fully OFF, 1: fully ON)
        dt = 1.0 / n_points
        
        for i, v in enumerate(V):
            if v > 0.3:
                dx = self.eta * (v - 0.3) * (1 - x) * dt * 5
                x = min(x + dx, 1.0)
            elif v < -0.3:
                dx = self.eta * (abs(v) - 0.3) * x * dt * 5
                x = max(x - dx, 0.0)
            
            R = self.R_max - x * (self.R_max - self.R_min)
            I[i] = (v / R) * (1 + 0.1 * np.sinh(v))
            
        return V, I


def simulate_stp(n_pulses=15, pulse_interval=50, tau_decay=200, delta_g=0.08, T=1200):
    """Simulates Short-Term Potentiation (STP) where conductance decays to baseline."""
    dt = 0.5
    t = np.arange(0, T, dt)
    g = np.zeros(len(t))
    g_baseline = 0.2
    g[0] = g_baseline
    
    pulse_times = [100 + i * pulse_interval for i in range(n_pulses)]
    pulse_indices = set(int(pt / dt) for pt in pulse_times if pt < T)
    
    for i in range(1, len(t)):
        dg = -(g[i-1] - g_baseline) / tau_decay
        g[i] = g[i-1] + dg * dt
        if i in pulse_indices:
            g[i] += delta_g
        g[i] = np.clip(g[i], 0, 1)
        
    return t, g, pulse_times


def simulate_std(n_pulses=15, pulse_interval=50, tau_recovery=200, delta_g=0.08, T=1200):
    """Simulates Short-Term Depression (STD) where conductance recovers to baseline."""
    dt = 0.5
    t = np.arange(0, T, dt)
    g = np.zeros(len(t))
    g_baseline = 0.8
    g[0] = g_baseline
    
    pulse_times = [100 + i * pulse_interval for i in range(n_pulses)]
    pulse_indices = set(int(pt / dt) for pt in pulse_times if pt < T)
    
    for i in range(1, len(t)):
        dg = -(g[i-1] - g_baseline) / tau_recovery
        g[i] = g[i-1] + dg * dt
        if i in pulse_indices:
            g[i] -= delta_g
        g[i] = np.clip(g[i], 0, 1)
        
    return t, g, pulse_times


def plot_memristive_devices(save_path=None):
    """Plots digital/analog memristor I-V curves and short-term plasticity (STP/STD)."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 150,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Memristive Device Characteristics\n(Nawrocki et al., IEEE TED 2016 — Fig. 5, Section III-IV)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    color_set = '#E63946'
    color_reset = '#457B9D'
    color_stp = '#2A9D8F'
    color_std = '#E76F51'
    
    # (a) Digital Memristive I-V curve
    digital = DigitalMemristor(R_on=1e3, R_off=1e5, V_set=0.8, V_reset=-0.6)
    V_dig, I_dig = digital.iv_curve(V_max=1.5)
    
    ax = axes[0, 0]
    n = len(V_dig)
    quarter = n // 4
    
    ax.plot(V_dig[:quarter], I_dig[:quarter]*1e3, color=color_set, linewidth=2, label='0→+V (SET)')
    ax.plot(V_dig[quarter:2*quarter], I_dig[quarter:2*quarter]*1e3, color=color_set, linewidth=2, alpha=0.6, linestyle='--')
    ax.plot(V_dig[2*quarter:3*quarter], I_dig[2*quarter:3*quarter]*1e3, color=color_reset, linewidth=2, label='0→-V (RESET)')
    ax.plot(V_dig[3*quarter:], I_dig[3*quarter:]*1e3, color=color_reset, linewidth=2, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (mA)')
    ax.set_title('(a) Digital Memristor I–V [Fig. 5(a)]', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(fontsize=9)
    
    ax.annotate('LRS\n(Low R)', xy=(1.0, 0.8), fontsize=9, color=color_set, fontweight='bold', ha='center')
    ax.annotate('HRS\n(High R)', xy=(-1.0, -0.005), fontsize=9, color=color_reset, fontweight='bold', ha='center')
    
    # (b) Analog Memristive I-V curve
    analog = AnalogMemristor(R_min=1e3, R_max=1e5, eta=1.0)
    V_ana, I_ana = analog.iv_curve(V_max=1.2)
    
    ax = axes[0, 1]
    n = len(V_ana)
    quarter = n // 4
    
    ax.plot(V_ana[:quarter], I_ana[:quarter]*1e3, color=color_set, linewidth=2, label='Gradual SET')
    ax.plot(V_ana[quarter:2*quarter], I_ana[quarter:2*quarter]*1e3, color=color_set, linewidth=2, alpha=0.6, linestyle='--')
    ax.plot(V_ana[2*quarter:3*quarter], I_ana[2*quarter:3*quarter]*1e3, color=color_reset, linewidth=2, label='Gradual RESET')
    ax.plot(V_ana[3*quarter:], I_ana[3*quarter:]*1e3, color=color_reset, linewidth=2, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (mA)')
    ax.set_title('(b) Analog Memristor I–V [Fig. 5(b)]', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(fontsize=9)
    
    ax.annotate('Multiple\nconductance\nlevels', xy=(0.75, 0.35), fontsize=8,
                color='gray', fontstyle='italic', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # (c) Short-Term Potentiation (STP)
    t_stp, g_stp, pulses_stp = simulate_stp()
    ax = axes[1, 0]
    ax.plot(t_stp, g_stp, color=color_stp, linewidth=2, label='Conductance')
    for pt in pulses_stp:
        ax.axvline(x=pt, color='gray', alpha=0.2, linewidth=1)
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Conductance')
    ax.set_title('(c) Short-Term Potentiation (STP)', fontweight='bold')
    ax.legend(fontsize=9)
    
    ax.annotate('Each pulse\nincreases G', xy=(400, 0.65), fontsize=9,
                color=color_stp, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color_stp),
                xytext=(550, 0.75))
    
    # (d) Short-Term Depression (STD)
    t_std, g_std, pulses_std = simulate_std()
    ax = axes[1, 1]
    ax.plot(t_std, g_std, color=color_std, linewidth=2, label='Conductance')
    for pt in pulses_std:
        ax.axvline(x=pt, color='gray', alpha=0.2, linewidth=1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Conductance')
    ax.set_title('(d) Short-Term Depression (STD)', fontweight='bold')
    ax.legend(fontsize=9)
    
    ax.annotate('Each pulse\ndecreases G', xy=(400, 0.35), fontsize=9,
                color=color_std, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color_std),
                xytext=(550, 0.25))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
        
    plt.show()
    return fig

if __name__ == "__main__":
    plot_memristive_devices(save_path='figures/fig3_memristive_devices.png')
