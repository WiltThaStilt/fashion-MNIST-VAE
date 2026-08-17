from encoder import Encoder
from reparameterization import Reparameterization
from decoder import Decoder
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, input_dim=16):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = Encoder(input_dim)
        self.reparameterization = Reparameterization(input_dim),
        self.decoder = Decoder(input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x) # Grabs the mean and the log-variance
        z = self.reparameterization(mu, logvar)
        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar