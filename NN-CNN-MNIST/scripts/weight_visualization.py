# codes to make visualization of your weights.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mynn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = nn.models.Model_MLP()
model.load_model(os.path.join(BASE_DIR, 'saved_models', 'best_model.pickle'))

mats = []
mats.append(model.layers[0].params['W'])
mats.append(model.layers[2].params['W'])

plt.figure()
plt.matshow(mats[1])
plt.xticks([])
plt.yticks([])

fig_path = os.path.join(BASE_DIR, 'results', 'weight_visualization.png')
plt.savefig(fig_path, dpi=150)
print(f'Weight visualization saved to {fig_path}')
