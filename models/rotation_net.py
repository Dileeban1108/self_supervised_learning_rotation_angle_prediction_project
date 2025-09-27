import torch.nn as nn
import torch.nn.functional as F

class RotationNet(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, feature_dim)  # learned features
        self.fc2 = nn.Linear(feature_dim, 4)     # 4 rotation classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (12, 12))
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(x))  # feature representation
        out = self.fc2(features)        # rotation classification
        return out, features
