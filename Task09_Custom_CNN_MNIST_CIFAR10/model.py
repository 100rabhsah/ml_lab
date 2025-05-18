import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(CustomCNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers will be initialized in forward pass
        self.fc1 = None
        self.fc2 = None
        self.dropout = nn.Dropout(0.5)
        
    def _create_fc_layers(self, x):
        # Calculate the size of flattened features
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        fc_input_size = x.size(1)
        
        # Create fully connected layers if not already created
        if self.fc1 is None:
            self.fc1 = nn.Linear(fc_input_size, 128).to(x.device)
            self.fc2 = nn.Linear(128, 10).to(x.device)
        
        return x
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Create and apply fully connected layers
        x = self._create_fc_layers(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 