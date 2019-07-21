import os
import sys

import numpy as np
import torch.nn as nn
from data.feature_data import NUM_FEATURES

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..'))
    sys.path.append(base_dir)


class FeatureModel(nn.Module):
    def __init__(self):
        super(FeatureModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features=NUM_FEATURES, out_features=4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4, out_features=8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8, out_features=8)])
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        outputs = list()
        output = x
        for layer in self.layers:
            output = layer(output)
            if isinstance(layer, nn.Linear):
                outputs.append(output)
        return np.array(outputs)


if __name__ == '__main__':
    raise NotImplementedError
