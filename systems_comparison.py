"""
Module 5: Neuromorphic Systems Comparison

Compiles and plots structural and efficiency metrics of different neuromorphic systems
(Neurogrid, TrueNorth, SpiNNaker, ROLLS, etc.) compared to biological reference systems.
Data is compiled from Table I of Nawrocki et al. (IEEE TED 2016).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Systems database compiling Table I specs
SYSTEMS = {
    'Neurogrid': {
        'neurons': 1e6,
        'synapses': 7e9,
        'power_W': 5.0,
        'technology': 'CMOS',
        'neuron_model': 'I&F',
        'institution': 'Stanford',
        'year': 2006,
        'category': 'hardware',
        'color': '#E63946',
    },
    'TrueNorth': {
        'neurons': 1e6,
        'synapses': 256e6,
        'power_W': 0.065,
        'technology': 'CMOS',
        'neuron_model': 'I&F',
        'institution': 'IBM',
        'year': 2014,
        'category': 'hardware',
        'color': '#457B9D',
    },
    'SpiNNaker': {
        'neurons': 1e9,
        'synapses': 1e12,
        'power_W': 90000,
        'technology': 'ARM + CMOS',
        'neuron_model': 'Various',
        'institution': 'Manchester',
        'year': 2012,
        'category': 'hardware',
        'color': '#2A9D8F',
    },
    'ROLLS': {
        'neurons': 256,
        'synapses': 128e3,
        'power_W': 0.004,
        'technology': 'CMOS',
        'neuron_model': 'I&F',
        'institution': 'ETH Zurich',
        'year': 2015,
        'category': 'hardware',
        'color': '#E9C46A',
    },
    'Blue Gene/P\n(Cat Brain)': {
        'neurons': 1e9,
        'synapses': 1e13,
        'power_W': 2.9e6,
        'technology': 'Software\non CPU',
        'neuron_model': 'Biophysical',
        'institution': 'IBM',
        'year': 2009,
        'category': 'software',
        'color': '#264653',
    },
    'PCM Network\n(IBM)': {
        'neurons': 948,
        'synapses': 165000,
        'power_W': 1.0,
        'technology': 'PCM +\nCMOS',
        'neuron_model': 'Perceptron',
        'institution': 'IBM',
        'year': 2015,
        'category': 'memristive',
        'color': '#E76F51',
    },
    'RPU Array\n(IBM)': {
        'neurons': 8192,
        'synapses': 16.7e6,
        'power_W': 0.5,
        'technology': 'Memristive\nCrossbar',
        'neuron_model': 'ANN',
        'institution': 'IBM',
        'year': 2016,
        'category': 'memristive',
        'color': '#F4A261',
    },
}

# Biological references
BIOLOGICAL = {
    'Fruit Fly': {
        'neurons': 1e5,
        'power_W': 1e-5,
        'color': '#8ECE7A',
    },
    'Human Brain': {
        'neurons': 86e9,
        'power_W': 20,
        'color': '#C77DFF',
    },
}

def plot_systems_comparison(save_path=None):
    """Generates comparison charts comparing neuron count, power, and energy efficiency."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 150,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Neuromorphic Systems Comparison\n(Nawrocki et al., IEEE TED 2016 — Table I, Section V)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    names = list(SYSTEMS.keys())
    colors = [SYSTEMS[n]['color'] for n in names]
    
    # (a) Neuron Count
    ax = axes[0, 0]
    neurons = [SYSTEMS[n]['neurons'] for n in names]
    bars = ax.barh(range(len(names)), neurons, color=colors, edgecolor='white', linewidth=1.5, height=0.7)
    ax.set_xscale('log')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Number of Neurons (log scale)')
    ax.set_title('(a) Neuron Count per System', fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, neurons)):
        if val >= 1e9:
            label = f'{val/1e9:.0f}B'
        elif val >= 1e6:
            label = f'{val/1e6:.0f}M'
        elif val >= 1e3:
            label = f'{val/1e3:.0f}K'
        else:
            label = f'{val:.0f}'
        ax.text(val * 1.5, i, label, va='center', fontsize=8, fontweight='bold')
        
    ax.axvline(x=86e9, color='#C77DFF', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(86e9, -0.7, 'Human\nBrain\n(86B)', fontsize=7, color='#C77DFF', ha='center', fontweight='bold')
    
    # (b) Power Consumption
    ax = axes[0, 1]
    power = [SYSTEMS[n]['power_W'] for n in names]
    bars = ax.barh(range(len(names)), power, color=colors, edgecolor='white', linewidth=1.5, height=0.7)
    ax.set_xscale('log')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Power Consumption (W, log scale)')
    ax.set_title('(b) Power Consumption', fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, power)):
        if val >= 1e6:
            label = f'{val/1e6:.1f} MW'
        elif val >= 1e3:
            label = f'{val/1e3:.0f} kW'
        elif val >= 1:
            label = f'{val:.1f} W'
        elif val >= 1e-3:
            label = f'{val*1e3:.0f} mW'
        else:
            label = f'{val*1e6:.0f} µW'
        ax.text(val * 1.5, i, label, va='center', fontsize=8, fontweight='bold')
        
    ax.axvline(x=20, color='#C77DFF', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(20, -0.7, 'Human Brain\n(20 W)', fontsize=7, color='#C77DFF', ha='center', fontweight='bold')
    
    # (c) Energy Efficiency: Neurons per Watt
    ax = axes[1, 0]
    efficiency = [SYSTEMS[n]['neurons'] / SYSTEMS[n]['power_W'] for n in names]
    bars = ax.barh(range(len(names)), efficiency, color=colors, edgecolor='white', linewidth=1.5, height=0.7)
    ax.set_xscale('log')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Neurons per Watt (log scale)')
    ax.set_title('(c) Energy Efficiency (Neurons/Watt)', fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, efficiency)):
        if val >= 1e9:
            label = f'{val/1e9:.1f}B/W'
        elif val >= 1e6:
            label = f'{val/1e6:.1f}M/W'
        elif val >= 1e3:
            label = f'{val/1e3:.0f}K/W'
        else:
            label = f'{val:.0f}/W'
        ax.text(val * 1.5, i, label, va='center', fontsize=8, fontweight='bold')
        
    brain_eff = 86e9 / 20
    ax.axvline(x=brain_eff, color='#C77DFF', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(brain_eff, -0.7, f'Human Brain\n({brain_eff/1e9:.1f}B/W)', fontsize=7, 
            color='#C77DFF', ha='center', fontweight='bold')
    
    # (d) Biological vs Artificial System Scatter Plot
    ax = axes[1, 1]
    for name in names:
        s = SYSTEMS[name]
        ax.scatter(s['power_W'], s['neurons'], color=s['color'], s=150, edgecolor='white', linewidth=1.5, zorder=5)
        offset_x = 1.5
        offset_y = 1.2
        ax.annotate(name.replace('\n', ' '), xy=(s['power_W'], s['neurons']),
                    xytext=(s['power_W'] * offset_x, s['neurons'] * offset_y),
                    fontsize=7, fontweight='bold', color=s['color'],
                    arrowprops=dict(arrowstyle='-', color=s['color'], alpha=0.3))
        
    for name, bio in BIOLOGICAL.items():
        ax.scatter(bio['power_W'], bio['neurons'], color=bio['color'], s=250, marker='*', edgecolor='white', linewidth=1.5, zorder=5)
        ax.annotate(name, xy=(bio['power_W'], bio['neurons']), fontsize=9, fontweight='bold', color=bio['color'],
                    xytext=(bio['power_W'] * 0.3, bio['neurons'] * 1.5))
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Power Consumption (W)')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('(d) Biological vs Artificial Systems', fontweight='bold')
    
    hw_patch = mpatches.Patch(color='#457B9D', label='Hardware neuromorphic')
    sw_patch = mpatches.Patch(color='#264653', label='Software simulation')
    mem_patch = mpatches.Patch(color='#E76F51', label='Memristive-based')
    bio_patch = mpatches.Patch(facecolor='#8ECE7A', label='Biological', edgecolor='white')
    ax.legend(handles=[hw_patch, sw_patch, mem_patch, bio_patch], fontsize=8, loc='lower right')
    
    ax.annotate('', xy=(1e-5, 1e11), xytext=(1e6, 1e2), arrowprops=dict(arrowstyle='->', color='green', alpha=0.3, linewidth=3))
    ax.text(1e-2, 1e8, 'Ideal\ndirection', fontsize=8, color='green', alpha=0.5, ha='center', fontstyle='italic', rotation=30)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
        
    plt.show()
    return fig

def print_table():
    """Prints a formatted text table matching Table I of the review paper."""
    print("\n" + "=" * 100)
    print("TABLE I: SUMMARY OF NEUROMORPHIC SYSTEMS IMPLEMENTATIONS")
    print("(Recreated from Nawrocki et al., IEEE TED 2016)")
    print("=" * 100)
    
    header = f"{'System':<20} {'Neurons':>12} {'Synapses':>12} {'Power':>12} {'Technology':<15} {'Model':<12}"
    print(header)
    print("-" * 100)
    
    for name, s in SYSTEMS.items():
        clean_name = name.replace('\n', ' ')
        
        if s['neurons'] >= 1e9:
            n_str = f"{s['neurons']/1e9:.0f}B"
        elif s['neurons'] >= 1e6:
            n_str = f"{s['neurons']/1e6:.0f}M"
        elif s['neurons'] >= 1e3:
            n_str = f"{s['neurons']/1e3:.0f}K"
        else:
            n_str = f"{s['neurons']:.0f}"
            
        if s['synapses'] >= 1e12:
            s_str = f"{s['synapses']/1e12:.0f}T"
        elif s['synapses'] >= 1e9:
            s_str = f"{s['synapses']/1e9:.0f}B"
        elif s['synapses'] >= 1e6:
            s_str = f"{s['synapses']/1e6:.0f}M"
        elif s['synapses'] >= 1e3:
            s_str = f"{s['synapses']/1e3:.0f}K"
        else:
            s_str = f"{s['synapses']:.0f}"
            
        if s['power_W'] >= 1e6:
            p_str = f"{s['power_W']/1e6:.1f} MW"
        elif s['power_W'] >= 1e3:
            p_str = f"{s['power_W']/1e3:.0f} kW"
        elif s['power_W'] >= 1:
            p_str = f"{s['power_W']:.1f} W"
        else:
            p_str = f"{s['power_W']*1e3:.0f} mW"
            
        tech = s['technology'].replace('\n', ' ')
        print(f"{clean_name:<20} {n_str:>12} {s_str:>12} {p_str:>12} {tech:<15} {s['neuron_model']:<12}")
        
    print("-" * 100)
    print(f"{'Human Brain':<20} {'86B':>12} {'100T':>12} {'20 W':>12} {'Biological':<15} {'Biophysical':<12}")
    print("=" * 100)

if __name__ == "__main__":
    print_table()
    plot_systems_comparison(save_path='figures/fig5_systems_comparison.png')
