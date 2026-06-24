"""
Module 4: Memristive Crossbar Neural Network

Simulates a single-layer perceptron built using a memristive crossbar array
to classify 3x3 pixel binary patterns, modeling the UCSB Al2O3/TiO2 device configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Define 3x3 pixel template patterns for classification classes
PATTERNS = {
    'T': np.array([
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    'L': np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]),
    'I': np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    'C': np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ]),
    'X': np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]),
    'O': np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]),
}

class MemristiveCrossbar:
    """
    Simulates a memristive crossbar circuit implementing a single-layer perceptron.
    Row inputs map to pixel voltages; columns compute weighted sum currents via Ohm's/Kirchhoff's laws.
    """
    
    def __init__(self, n_inputs=9, n_classes=6, G_min=0.1, G_max=1.0):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.G_min = G_min
        self.G_max = G_max
        
        # Initialize synaptic weights (crossbar conductances) randomly
        self.W = np.random.uniform(G_min, G_max, (n_inputs, n_classes))
        self.bias = np.zeros(n_classes)
    
    def forward(self, x):
        """Computes outputs through matrix multiplication with the conductance weights."""
        return np.dot(x, self.W) + self.bias
    
    def predict(self, x):
        """Winner-take-all output prediction."""
        return np.argmax(self.forward(x))
    
    def train(self, X_train, y_train, epochs=100, lr=0.1):
        """Trains the crossbar weights using a perceptron learning rule with clipped conductances."""
        loss_history = []
        accuracy_history = []
        
        for epoch in range(epochs):
            correct = 0
            total_loss = 0
            
            for x, y_true in zip(X_train, y_train):
                output = self.forward(x)
                y_pred = np.argmax(output)
                
                if y_pred == y_true:
                    correct += 1
                else:
                    # Update weights (conductances) and biases
                    self.W[:, y_true] += lr * x
                    self.W[:, y_pred] -= lr * x
                    self.bias[y_true] += lr
                    self.bias[y_pred] -= lr
                    
                    # Clip weights to physical device limits
                    self.W = np.clip(self.W, self.G_min, self.G_max)
                
                total_loss += (0 if y_pred == y_true else 1)
            
            accuracy = correct / len(y_train) * 100
            loss_history.append(total_loss / len(y_train))
            accuracy_history.append(accuracy)
            
        return loss_history, accuracy_history
    
    def test_with_noise(self, patterns, class_names, noise_levels):
        """Evaluates accuracy with random pixel-flip noise applied to the inputs."""
        n_trials = 200
        accuracies = []
        
        for noise_level in noise_levels:
            correct = 0
            total = 0
            
            for class_idx, name in enumerate(class_names):
                pattern = patterns[name].flatten().astype(float)
                
                for _ in range(n_trials):
                    noisy = pattern.copy()
                    flip_mask = np.random.random(9) < noise_level
                    noisy[flip_mask] = 1 - noisy[flip_mask]
                    
                    pred = self.predict(noisy)
                    if pred == class_idx:
                        correct += 1
                    total += 1
            
            accuracies.append(correct / total * 100)
            
        return accuracies


def plot_crossbar_network(save_path=None):
    """Generates the training, weight, and noise analysis plots."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 150,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Memristive Crossbar Neural Network\n(Based on UCSB Al₂O₃/TiOₓ crossbar — ref [83], Fig. 6)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    class_names = list(PATTERNS.keys())
    n_classes = len(class_names)
    
    X_train = []
    y_train = []
    
    for class_idx, name in enumerate(class_names):
        pattern = PATTERNS[name].flatten().astype(float)
        # Clean patterns
        for _ in range(20):
            X_train.append(pattern)
            y_train.append(class_idx)
        # Slightly noisy patterns
        for _ in range(30):
            noisy = pattern.copy()
            flip_mask = np.random.random(9) < 0.1
            noisy[flip_mask] = 1 - noisy[flip_mask]
            X_train.append(noisy)
            y_train.append(class_idx)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    np.random.seed(42)
    crossbar = MemristiveCrossbar(n_inputs=9, n_classes=n_classes)
    loss_history, accuracy_history = crossbar.train(X_train, y_train, epochs=80, lr=0.05)
    
    # (a) 3x3 pattern visualization
    ax = axes[0, 0]
    ax.set_axis_off()
    ax.set_title('(a) 3×3 Pixel Pattern Classes', fontweight='bold')
    for i, name in enumerate(class_names):
        inset_ax = fig.add_axes([0.07 + i * 0.065, 0.7, 0.05, 0.1])
        inset_ax.imshow(PATTERNS[name], cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)
        inset_ax.set_title(f'"{name}"', fontsize=9, fontweight='bold')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_edgecolor('#457B9D')
            spine.set_linewidth(2)
            
    # (b) Training convergence curve
    ax = axes[0, 1]
    color_acc = '#2A9D8F'
    color_loss = '#E63946'
    
    ax.plot(accuracy_history, color=color_acc, linewidth=2, label='Accuracy (%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)', color=color_acc)
    ax.tick_params(axis='y', labelcolor=color_acc)
    ax.set_ylim(0, 105)
    
    ax2 = ax.twinx()
    ax2.plot(loss_history, color=color_loss, linewidth=2, alpha=0.7, label='Loss')
    ax2.set_ylabel('Loss', color=color_loss)
    ax2.tick_params(axis='y', labelcolor=color_loss)
    
    ax.set_title('(b) Training Convergence', fontweight='bold')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')
    
    final_acc = accuracy_history[-1]
    ax.text(0.05, 0.15, f'Final accuracy: {final_acc:.1f}%\n(Paper reports ~83%\nfor hardware)',
            transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # (c) Conductance weight matrix heatmap
    ax = axes[1, 0]
    im = ax.imshow(crossbar.W, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Output Class')
    ax.set_ylabel('Input Pixel')
    ax.set_title('(c) Learned Conductance Matrix G', fontweight='bold')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(9))
    ax.set_yticklabels([f'px({r},{c})' for r in range(3) for c in range(3)], fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Conductance (a.u.)', fontsize=9)
    
    # (d) Classification accuracy vs noise level
    noise_levels = np.arange(0, 0.55, 0.05)
    accuracies = crossbar.test_with_noise(PATTERNS, class_names, noise_levels)
    
    ax = axes[1, 1]
    ax.plot(noise_levels * 100, accuracies, 'o-', color='#264653', linewidth=2, 
            markersize=6, markerfacecolor='#E9C46A', markeredgecolor='#264653')
    ax.axhline(y=83, color='gray', linestyle='--', alpha=0.5, 
               label='Hardware benchmark (~83%, ref [83])')
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title('(d) Accuracy vs Input Noise', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    
    ax.axvspan(0, 15, alpha=0.05, color='green', label='Low noise')
    ax.axvspan(15, 35, alpha=0.05, color='orange')
    ax.axvspan(35, 55, alpha=0.05, color='red')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
        
    plt.show()
    return fig

if __name__ == "__main__":
    plot_crossbar_network(save_path='figures/fig4_crossbar_network.png')
