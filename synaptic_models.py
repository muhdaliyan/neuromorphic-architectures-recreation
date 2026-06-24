"""
Module 2: Synaptic Dynamics and Plasticity

Implements:
  1. Current Mirror Integrator Synapse model (Bartolozzi & Indiveri).
  2. Spike-Timing-Dependent Plasticity (STDP) learning rule.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class CurrentMirrorSynapse:
    """
    Excitatory synaptic circuit simulating temporal integration of input charge.
    Each incoming spike adds a fixed charge, which then decays exponentially.
    """
    
    def __init__(self, weight=1.0, tau_syn=20.0, C=1.0, delta_q=0.5):
        self.weight = weight
        self.tau_syn = tau_syn
        self.C = C
        self.delta_q = delta_q
    
    def simulate(self, spike_times, T=100.0, dt=0.1):
        """Simulates charge accumulation and output current over time."""
        t = np.arange(0, T, dt)
        Q = np.zeros(len(t))
        I_EX = np.zeros(len(t))
        
        spike_indices = [int(st / dt) for st in spike_times if st < T]
        
        for i in range(1, len(t)):
            # Leak decay
            dQdt = -Q[i-1] / self.tau_syn
            Q[i] = Q[i-1] + dQdt * dt
            
            # Spike deposition
            if i in spike_indices:
                Q[i] += self.delta_q
            
            I_EX[i] = self.weight * Q[i] / self.C
            
        return t, Q, I_EX


class STDP:
    """
    Spike-Timing-Dependent Plasticity (STDP) learning rule.
    Strengthens (LTP) or weakens (LTD) weights based on the relative timing of spikes.
    """
    
    def __init__(self, A_plus=0.01, A_minus=0.012, tau_plus=20.0, tau_minus=20.0):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
    
    def weight_change(self, delta_t):
        """Calculates synaptic weight change for given spike timing difference."""
        if isinstance(delta_t, np.ndarray):
            dw = np.where(
                delta_t > 0,
                self.A_plus * np.exp(-np.abs(delta_t) / self.tau_plus),
                -self.A_minus * np.exp(-np.abs(delta_t) / self.tau_minus)
            )
            dw[delta_t == 0] = 0.0
            return dw
        else:
            if delta_t > 0:
                return self.A_plus * np.exp(-abs(delta_t) / self.tau_plus)
            elif delta_t < 0:
                return -self.A_minus * np.exp(-abs(delta_t) / self.tau_minus)
            else:
                return 0.0
    
    def evolve_weight(self, w0, pre_spikes, post_spikes, w_min=0.0, w_max=1.0):
        """Evolves synaptic weight over repeated pre/post spike pairings."""
        w = w0
        weights = [w]
        dw_list = []
        
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                delta_t = t_post - t_pre
                if abs(delta_t) < 5 * max(self.tau_plus, self.tau_minus):
                    dw = self.weight_change(delta_t)
                    w = np.clip(w + dw, w_min, w_max)
                    dw_list.append(dw)
                    weights.append(w)
                    
        return weights, dw_list


def plot_synaptic_models(save_path=None):
    """Plots synapse simulation results and STDP curves."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 150,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Synaptic Dynamics and Plasticity\n(Nawrocki et al., IEEE TED 2016 — Fig. 2(b), Section III)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    color_charge = '#2A9D8F'
    color_current = '#E76F51'
    color_ltp = '#457B9D'
    color_ltd = '#E63946'
    
    # (a) Current Mirror Synapse: Charge Accumulation
    spike_times = [10, 15, 20, 25, 35, 45, 55, 70]
    synapse = CurrentMirrorSynapse(weight=1.0, tau_syn=15.0, delta_q=0.5)
    t, Q, I_EX = synapse.simulate(spike_times, T=100.0, dt=0.1)
    
    ax = axes[0, 0]
    ax.plot(t, Q, color=color_charge, linewidth=2, label='Capacitor charge Q')
    for st in spike_times:
        ax.axvline(x=st, color='gray', alpha=0.3, linewidth=1, linestyle='--')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Charge Q (a.u.)')
    ax.set_title('(a) Current Mirror Synapse: Charge Build-up', fontweight='bold')
    ax.scatter(spike_times, [0.05]*len(spike_times), color=color_current, 
               marker='|', s=200, linewidth=2, label='Input spikes (V_S)', zorder=5)
    ax.legend(fontsize=9)
    
    # (b) Current Mirror Synapse: Output Current
    ax = axes[0, 1]
    ax.plot(t, I_EX, color=color_current, linewidth=2, label='Excitatory current I_EX')
    ax.fill_between(t, 0, I_EX, alpha=0.15, color=color_current)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Output Current I_EX (a.u.)')
    ax.set_title('(b) Excitatory Output Current', fontweight='bold')
    ax.legend(fontsize=9)
    
    ax.annotate('"With each input pulse, a fixed\namount of charge is stored on\nthe capacitor, and I_EX increases"',
                xy=(0.55, 0.75), xycoords='axes fraction',
                fontsize=8, fontstyle='italic', color='gray',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # (c) STDP Learning Window
    stdp = STDP(A_plus=0.01, A_minus=0.012, tau_plus=20.0, tau_minus=20.0)
    delta_t = np.linspace(-60, 60, 1000)
    delta_t = delta_t[delta_t != 0]
    dw = stdp.weight_change(delta_t)
    
    ax = axes[1, 0]
    pos_mask = delta_t > 0
    neg_mask = delta_t < 0
    ax.fill_between(delta_t[pos_mask], 0, dw[pos_mask], alpha=0.2, color=color_ltp)
    ax.fill_between(delta_t[neg_mask], 0, dw[neg_mask], alpha=0.2, color=color_ltd)
    ax.plot(delta_t[pos_mask], dw[pos_mask], color=color_ltp, linewidth=2, label='LTP (potentiation)')
    ax.plot(delta_t[neg_mask], dw[neg_mask], color=color_ltd, linewidth=2, label='LTD (depression)')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Δt = t_post − t_pre (ms)')
    ax.set_ylabel('Weight Change Δw')
    ax.set_title('(c) STDP Learning Window', fontweight='bold')
    ax.legend(fontsize=9)
    
    ax.text(30, 0.005, 'Pre → Post\n(Strengthen)', fontsize=8, ha='center', color=color_ltp)
    ax.text(-30, -0.006, 'Post → Pre\n(Weaken)', fontsize=8, ha='center', color=color_ltd)
    
    # (d) Weight Evolution Under Repeated Pairings
    n_pairings = 60
    pre_spikes_ltp = np.arange(0, n_pairings * 50, 50)
    post_spikes_ltp = pre_spikes_ltp + 10
    w_ltp, _ = stdp.evolve_weight(0.5, pre_spikes_ltp, post_spikes_ltp)
    
    pre_spikes_ltd = np.arange(0, n_pairings * 50, 50)
    post_spikes_ltd = pre_spikes_ltd - 10
    w_ltd, _ = stdp.evolve_weight(0.5, pre_spikes_ltd, post_spikes_ltd)
    
    ax = axes[1, 1]
    ax.plot(range(len(w_ltp)), w_ltp, color=color_ltp, linewidth=2, label='LTP (Δt = +10 ms)')
    ax.plot(range(len(w_ltd)), w_ltd, color=color_ltd, linewidth=2, label='LTD (Δt = -10 ms)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Initial weight')
    ax.set_xlabel('Spike Pairing Number')
    ax.set_ylabel('Synaptic Weight w')
    ax.set_title('(d) Weight Evolution Under Repeated STDP', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
        
    plt.show()
    return fig

if __name__ == "__main__":
    plot_synaptic_models(save_path='figures/fig2_synaptic_models.png')
