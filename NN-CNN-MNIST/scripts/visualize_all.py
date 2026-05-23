# Generate all visualization figures after training
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mynn as nn
from mynn.metrics import confusion_matrix, misclassified_indices

import numpy as np
import gzip
from struct import unpack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Load test data
test_images_path = os.path.join(BASE_DIR, 'data', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(BASE_DIR, 'data', 'MNIST', 't10k-labels-idx1-ubyte.gz')

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    raw = f.read()
    test_imgs_flat = np.frombuffer(raw, dtype=np.uint8).reshape(num, 28 * 28).astype(np.float64) / 255.0
    test_imgs_cnn = np.frombuffer(raw, dtype=np.uint8).reshape(num, 1, 28, 28).astype(np.float64) / 255.0

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)


def plot_confusion_matrix(cm, title, save_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=8, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved {save_path}')


def plot_misclassified(imgs, true_labels, pred_labels, mis_idx, title, save_path, n=25):
    """Plot misclassified examples."""
    n = min(n, len(mis_idx))
    rows = int(np.ceil(n / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(10, 2 * rows))
    axes = axes.reshape(-1)
    for k in range(n):
        idx = mis_idx[k]
        axes[k].imshow(imgs[idx].reshape(28, 28), cmap='gray')
        axes[k].set_title(f'T={true_labels[idx]} P={pred_labels[idx]}', fontsize=9)
        axes[k].axis('off')
    for k in range(n, len(axes)):
        axes[k].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved {save_path}')


def plot_mlp_weights(model, save_path):
    """Visualize MLP first layer weights as 28x28 images."""
    W1 = model.layers[0].params['W']  # [784, 600]
    n_neurons = min(25, W1.shape[1])
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.reshape(-1)
    for i in range(n_neurons):
        w = W1[:, i].reshape(28, 28)
        axes[i].imshow(w, cmap='RdBu', interpolation='nearest')
        axes[i].axis('off')
        axes[i].set_title(f'Neuron {i+1}', fontsize=8)
    for i in range(n_neurons, 25):
        axes[i].axis('off')
    plt.suptitle('MLP First Layer Weights (25/600 neurons)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved {save_path}')


def plot_cnn_kernels(model, save_path_conv1, save_path_conv2):
    """Visualize CNN convolution kernels."""
    # Conv1: [8, 1, 3, 3]
    W1 = model.layers[0].params['W']
    out_c, in_c, k, _ = W1.shape
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    axes = axes.reshape(-1)
    for i in range(out_c):
        axes[i].imshow(W1[i, 0, :, :], cmap='RdBu', interpolation='nearest')
        axes[i].axis('off')
        axes[i].set_title(f'K{i+1}')
    for i in range(out_c, 8):
        axes[i].axis('off')
    plt.suptitle('Conv1 Kernels (8 x 1x3x3)')
    plt.tight_layout()
    plt.savefig(save_path_conv1, dpi=150)
    plt.close()
    print(f'Saved {save_path_conv1}')

    # Conv2: [16, 8, 3, 3] - show first channel only
    W2 = model.layers[2].params['W']
    out_c2, in_c2, k2, _ = W2.shape
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.reshape(-1)
    for i in range(out_c2):
        axes[i].imshow(W2[i, 0, :, :], cmap='RdBu', interpolation='nearest')
        axes[i].axis('off')
        axes[i].set_title(f'K{i+1} (ch0)')
    for i in range(out_c2, 16):
        axes[i].axis('off')
    plt.suptitle('Conv2 Kernels (16, showing channel 0)')
    plt.tight_layout()
    plt.savefig(save_path_conv2, dpi=150)
    plt.close()
    print(f'Saved {save_path_conv2}')


# ==================== MLP Visualizations ====================
print("=" * 50)
print("Generating MLP visualizations...")
print("=" * 50)

mlp = nn.models.Model_MLP()
mlp.load_model(os.path.join(BASE_DIR, 'saved_models', 'mlp_model.pickle'))

# MLP Confusion Matrix
logits_mlp = mlp(test_imgs_flat)
cm_mlp = confusion_matrix(logits_mlp, test_labs)
plot_confusion_matrix(cm_mlp, 'MLP Confusion Matrix (Test Set)',
                      os.path.join(RESULTS_DIR, 'mlp_confusion_matrix.png'))

# MLP Misclassified
pred_labels_mlp = np.argmax(logits_mlp, axis=-1)
mis_mlp = misclassified_indices(logits_mlp, test_labs)
plot_misclassified(test_imgs_flat, test_labs, pred_labels_mlp, mis_mlp,
                   'MLP Misclassified Examples (T=True, P=Predicted)',
                   os.path.join(RESULTS_DIR, 'mlp_misclassified.png'))

# MLP Weights
plot_mlp_weights(mlp, os.path.join(RESULTS_DIR, 'mlp_weights.png'))


# ==================== CNN Visualizations ====================
print("=" * 50)
print("Generating CNN visualizations...")
print("=" * 50)

# Try 5epoch model first, fall back to best_model
cnn_model_path = os.path.join(BASE_DIR, 'saved_models', 'cnn_5epoch.pickle')
if not os.path.exists(cnn_model_path):
    cnn_model_path = os.path.join(BASE_DIR, 'saved_models', 'best_model.pickle')

cnn = nn.models.Model_CNN(in_channels=1, num_classes=10)
cnn.load_model(cnn_model_path)

# CNN Confusion Matrix
logits_cnn = cnn(test_imgs_cnn)
cm_cnn = confusion_matrix(logits_cnn, test_labs)
plot_confusion_matrix(cm_cnn, 'CNN Confusion Matrix (Test Set)',
                      os.path.join(RESULTS_DIR, 'cnn_confusion_matrix.png'))

# CNN Kernels
plot_cnn_kernels(cnn,
                 os.path.join(RESULTS_DIR, 'cnn_kernels_conv1.png'),
                 os.path.join(RESULTS_DIR, 'cnn_kernels_conv2.png'))

print("=" * 50)
print("All visualizations complete!")
print("=" * 50)
