# MLP training script with test evaluation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mynn as nn
from tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib
matplotlib.use('Agg')  # headless server support
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_images_path = os.path.join(BASE_DIR, 'data', 'MNIST', 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(BASE_DIR, 'data', 'MNIST', 'train-labels-idx1-ubyte.gz')

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open(os.path.join(BASE_DIR, 'idx.pickle'), 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
optimizer = nn.optim.SGD(init_lr=0.06, model=linear_model)
scheduler = nn.schedulers.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.layers.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metrics.accuracy, loss_fn, scheduler=scheduler)

save_dir = os.path.join(BASE_DIR, 'saved_models')
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=save_dir)

# Test on full test set
test_images_path = os.path.join(BASE_DIR, 'data', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(BASE_DIR, 'data', 'MNIST', 't10k-labels-idx1-ubyte.gz')
with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)
test_imgs = test_imgs / test_imgs.max()

logits = linear_model(test_imgs)
test_acc = nn.metrics.accuracy(logits, test_labs)
print(f'MLP Test accuracy: {test_acc:.4f}')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

# save figure instead of plt.show() for headless server
fig_path = os.path.join(BASE_DIR, 'results', 'training_curve.png')
plt.savefig(fig_path, dpi=150)
print(f'Training curve saved to {fig_path}')
