from encoder import Encoder
from reparameterization import Reparameterization
from decoder import Decoder
import torch.nn as nn
import torch


class VAE(nn.Module):

    def __init__(self, input_dim=16):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = Encoder(input_dim)
        self.reparameterization = Reparameterization(input_dim)
        self.decoder = Decoder(input_dim)

    def forward(self, x, sample=True):
        mu, log_var = self.encoder.forward(x) # Grabs the mean and the log-variance
        logvar = torch.clamp(
            log_var,
            min=-10.0,
            max=10.0,
        )
        if sample:
            z = self.reparameterization.forward(mu, logvar)
        else:
            z = mu
        reconstruction = self.decoder.forward(z)

        return reconstruction, mu, logvar