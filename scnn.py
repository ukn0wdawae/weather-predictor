pip install s2cnn

import torch
import torch.nn as nn
from s2cnn import S2Convolution, SO3Convolution

class SphericalCNN(nn.Module):
    def __init__(self):
        super(SphericalCNN, self).__init__()
        # Replace Conv2D with S2Convolution
        self.s2conv1 = S2Convolution(nfeature_in=1, nfeature_out=32, b_in=30, b_out=20)
        self.s2conv2 = S2Convolution(nfeature_in=32, nfeature_out=64, b_in=20, b_out=10)
        self.s2conv3 = S2Convolution(nfeature_in=64, nfeature_out=128, b_in=10, b_out=5)
        self.s2conv4 = S2Convolution(nfeature_in=128, nfeature_out=256, b_in=5, b_out=3)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 1)

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Forward pass through the spherical convolutions and pooling
        x = self.pool(nn.functional.relu(self.s2conv1(x)))
        x = self.pool(nn.functional.relu(self.s2conv2(x)))
        x = self.pool(nn.functional.relu(self.s2conv3(x)))
        x = self.pool(nn.functional.relu(self.s2conv4(x)))

        # Flatten the features for the fully connected layers
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate the model
model = SphericalCNN()

# Define the loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(model)
