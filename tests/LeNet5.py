import torch
import torch.nn as nn
import torch.nn.functional as F
from semifield import SemiConv2d

class SemiLeNet5(nn.Module):
    def __init__(self, num_classes=10, semi_field="MaxPlus", kernel_size=5, padding_mode=None):
        super(SemiLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)   # 28x28 -> 24x24
        self.pool1 = SemiConv2d(20, 20, semifield_type=semi_field, kernel_size=kernel_size, stride=2, alpha=10, groups=20, padding_mode=padding_mode, dtype=torch.float32)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)  # 12x12 -> 8x8
        self.pool2 = SemiConv2d(50, 50, semifield_type=semi_field, kernel_size=kernel_size, stride=2, alpha=10, groups=50, padding_mode=padding_mode, dtype=torch.float32)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # 1x28x28 → 20x24x24
        x = self.pool1(x)               # → 20x12x12
        x = F.relu(self.conv2(x))      # → 50x8x8
        x = self.pool2(x)               # → 50x4x4
        x = x.view(-1, 50 * 4 * 4)     # Flatten to 800
        x = F.relu(self.fc1(x))        # → 500
        x = self.fc2(x)                # → 10 logits
        return x                       # Raw logits
    
class StandardLeNet5(nn.Module):
    def __init__(self, num_classes=10, k=2, p=0):
        super(StandardLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)   # 28x28 -> 24x24
        self.pool = nn.MaxPool2d(kernel_size=k, stride=2, padding=p)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)  # 12x12 -> 8x8
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # 1x28x28 → 20x24x24
        x = self.pool(x)               # → 20x12x12
        x = F.relu(self.conv2(x))      # → 50x8x8
        x = self.pool(x)               # → 50x4x4
        x = x.view(-1, 50 * 4 * 4)     # Flatten to 800
        x = F.relu(self.fc1(x))        # → 500
        x = self.fc2(x)                # → 10 logits
        return x                       # Raw logits