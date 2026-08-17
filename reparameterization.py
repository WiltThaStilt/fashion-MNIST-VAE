import torch
import torch.nn as nn


class Reparameterization(nn.Module):


    def __init__(self, input_dim=16):
        super().__init__()
        self.input_dim = input_dim


    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z
