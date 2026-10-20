"""Small scratch-trained CNN suitable for a few hundred rendered examples."""

from torch import nn


class TinyInspectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(24 * 4 * 4, 32),
                                        nn.ReLU(), nn.Linear(32, 1))

    def forward(self, images):
        return self.classifier(self.features(images)).squeeze(1)
