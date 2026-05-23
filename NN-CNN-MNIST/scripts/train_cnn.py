import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mynn as nn
from tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(309)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_images_path = os.path.join(BASE_DIR, 'data', 'MNIST', 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(BASE_DIR, 'data', 'MNIST', 'train-labels-idx1-ubyte.gz')

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28).astype(np.float64)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

train_imgs = train_imgs.reshape(num, 1, 28, 28)

idx = np.random.permutation(np.arange(num))
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:1000]
valid_labs = train_labs[:1000]
train_imgs = train_imgs[1000:]
train_labs = train_labs[1000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

cnn_model = nn.models.Model_CNN(in_channels=1, num_classes=10)
optimizer = nn.optim.SGD(init_lr=0.01, model=cnn_model)
loss_fn = nn.layers.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metrics.accuracy, loss_fn, batch_size=32)

save_dir = os.path.join(BASE_DIR, 'saved_models')
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=5, log_iters=100, save_dir=save_dir)

# Test on full test set
test_images_path = os.path.join(BASE_DIR, 'data', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(BASE_DIR, 'data', 'MNIST', 't10k-labels-idx1-ubyte.gz')
with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28).astype(np.float64)
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)
test_imgs = test_imgs.reshape(num, 1, 28, 28) / test_imgs.max()

logits = cnn_model(test_imgs)
test_acc = nn.metrics.accuracy(logits, test_labs)
print(f'CNN Test accuracy: {test_acc:.4f}')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
fig_path = os.path.join(BASE_DIR, 'results', 'cnn_training_curve.png')
plt.savefig(fig_path, dpi=150)
print(f'CNN training curve saved to {fig_path}')
