import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    Value network for estimating state value V(s)
    """

    def __init__(self, obs_dim, hidden_size=128, num_layers=2, activation="ReLU"):
        super().__init__()
        self.obs_dim = obs_dim

        # Build hidden layers
        layers = []
        input_size = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "elu":
                layers.append(nn.ELU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            input_size = hidden_size

        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, 1)  # scalar state value

    def forward(self, x):
        x = self.net(x)
        value = self.output(x)
        return value
