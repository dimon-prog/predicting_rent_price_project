import torch
import torch.nn as nn



class PricePredictor(nn.Module):
    def __init__(self, input_features: int):
        super(PricePredictor, self).__init__()
        hidden_size = [256, 64]
        layers = []
        prev_size = input_features
        for layer in hidden_size:
            layers.extend([nn.Linear(prev_size, layer),
            #nn.BatchNorm1d(layer),
            nn.ReLU(),
            nn.Dropout(0.3)])
            prev_size = layer

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
