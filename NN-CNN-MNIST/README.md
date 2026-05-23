# Project 1: Neural Network and CNN from Scratch

Author: Huang Zhu (23307140048)

---

## Overview

Handwritten digit classification on MNIST using a neural network framework built **entirely from scratch with NumPy**. No deep learning frameworks (PyTorch, TensorFlow, etc.) are used for any operators. All core components — Linear, Conv2D, CrossEntropyLoss, SGD, Momentum — are hand-implemented.

### Results

|  | MLP | CNN |
|---|---|---|
| Architecture | 784→600→ReLU→10 | Conv(1,8)→ReLU→Conv(8,16)→ReLU→FC(10) |
| Parameters | 477K | 93K |
| Best Val Accuracy | 93.67% | 89.90% |
| Test Accuracy | 93.85% | 90.32% |
| Training Time (CPU) | ~0.5 h | ~2.2 h |

---

## Project Structure

```
NN-CNN-MNIST/
├── mynn/                   # Neural network framework
│   ├── layers/             # Layer, Linear, conv2D, ReLU, Flatten, CrossEntropyLoss
│   ├── models/             # Model_MLP, Model_CNN
│   ├── optim/              # SGD, MomentGD
│   ├── schedulers/         # StepLR, MultiStepLR, ExponentialLR
│   ├── metrics.py          # accuracy, confusion_matrix, misclassified_indices
│   └── runner.py           # Training loop (RunnerM)
├── scripts/
│   ├── train_mlp.py        # MLP training (with test evaluation)
│   ├── train_cnn.py        # CNN training (with test evaluation)
│   ├── visualize_all.py    # Generate all figures (confusion matrix, weights, kernels, etc.)
│   ├── test.py             # Evaluate a saved model on test set
│   ├── weight_visualization.py
│   └── hyperparameter_search.py
├── tools/
│   └── plot.py             # Loss/accuracy curve plotting
├── data/MNIST/             # Place MNIST .gz files here (see below)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `numpy>=1.26.0`, `matplotlib>=3.8.0`, `tqdm>=4.66.0` (verified with numpy 1.26.4 and 2.2.6).

### 2. Download MNIST

Download the 4 MNIST files from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) and place them in `data/MNIST/`:

```
data/MNIST/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

### 3. Train models

```bash
# MLP baseline (~0.5 hours on CPU)
python scripts/train_mlp.py

# CNN model (~2.2 hours on CPU)
python scripts/train_cnn.py
```

Training logs and curves are saved to `results/`. Models are saved to `saved_models/`.

### 4. Generate visualizations

```bash
python scripts/visualize_all.py
```

Generates: confusion matrices, MLP weight visualization, CNN kernel visualization, misclassified examples.

### 5. Evaluate a saved model

```bash
# Edit scripts/test.py to point to your model, then:
python scripts/test.py
```

---

## Implementation Details

### Key Technical Contributions

- **im2col vectorized Conv2D**: Replaced 4-level nested Python loops with matrix multiplication via BLAS, achieving ~1000× speedup (from ~27 days estimated to ~2.2 hours actual).
- **He initialization**: Prevents dying ReLU by scaling initial weights as W ~ N(0, √(2/fan_in)).
- **Numerical stability**: Softmax stabilized by subtracting max before exponentiation; cross-entropy gradient simplified to (ŷ - y)/N.
- **Modular framework**: Clean separation into layers, models, optim, and schedulers.

### Reproducibility

- Fixed random seed: `np.random.seed(309)`
- CPU-only, no GPU nondeterminism
- Verified consistent results across two independent runs

### Hyperparameters

|  | MLP | CNN |
|---|---|---|
| Optimizer | SGD (lr=0.06) | SGD (lr=0.01) |
| LR Schedule | MultiStepLR (γ=0.5) | None |
| Weight Decay | λ = 1e-4 | None |
| Weight Init | N(0,1) | He Normal |
| Batch Size | 32 | 32 |
| Epochs | 5 | 5 |
| Validation Set | 10,000 | 1,000 |

---

## Model Weights

Trained model weights are available at: *（https://drive.google.com/drive/folders/1yXurKEMRW1YmgWXPL1tYQT4nq5c5LvzZ?usp=drive_link)*

- `mlp_model.pickle` — MLP (Test 93.85%)
- `cnn_5epoch.pickle` — CNN (Test 90.32%)

---

## Requirements Compliance

- All operators (Linear, Conv2D, CrossEntropyLoss) implemented manually in NumPy
- No deep learning frameworks used for network operations
- CPU-only training
- MNIST dataset only (no external data)
- Results verified by full re-run on cloud server
