import torch
import torch.nn as nn

class LISYNet(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)