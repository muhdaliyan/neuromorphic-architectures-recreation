# Neuromorphic Architectures Recreation

Computational recreation of biological neuron models, synaptic dynamics, memristive device models, and crossbar neural networks described in:
> **A Mini Review of Neuromorphic Architectures and Implementations**  
> Nawrocki et al., *IEEE Transactions on Electron Devices* (2016)

---

## 📂 Project Structure

The project is structured into five simulation modules plus a master runner:

1.  **`neuron_models.py`**: Compares the biophysical Hodgkin-Huxley model and the hardware-friendly Leaky Integrate-and-Fire (LIF) model.
2.  **`synaptic_models.py`**: Simulates current-mirror integrator synapses and Spike-Timing-Dependent Plasticity (STDP) learning rules.
3.  **`memristive_devices.py`**: Models digital (bistable) and analog (multi-level) memristor I-V curves, including short-term plasticity (STP/STD).
4.  **`crossbar_network.py`**: Implements a single-layer memristive perceptron classifier for 3×3 pixel pattern recognition (achieving 90% classification accuracy).
5.  **`systems_comparison.py`**: Evaluates efficiency metrics of major neuromorphic platforms (TrueNorth, SpiNNaker, Neurogrid, ROLLS) vs. the human brain.
6.  **`run_all.py`**: Master execution script to run all simulations and regenerate all figures.

All simulation outputs and generated plots are stored in the `figures/` directory.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed, then install the required dependencies:
```bash
pip install numpy scipy matplotlib
```

### Running Simulations

Run the master script to execute all simulations and regenerate the figures:
```bash
python run_all.py
```

---

## 📈 Recreated Results & Figures

*   **Figure 1 (`fig1_neuron_models.png`)**: Membrane voltage potentials and gating probabilities.
*   **Figure 2 (`fig2_synaptic_models.png`)**: Synaptic integration and STDP weight changes over repeated spike pairings.
*   **Figure 3 (`fig3_memristive_devices.png`)**: Memristor hysteresis loops and STP/STD conductance modulation.
*   **Figure 4 (`fig4_crossbar_network.png`)**: Perceptron training convergence and noise-robustness evaluations.
*   **Figure 5 (`fig5_systems_comparison.png`)**: Energy efficiency and scaling comparison of neuromorphic platforms.
