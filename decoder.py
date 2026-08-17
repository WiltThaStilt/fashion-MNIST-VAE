from torch import nn


class Decoder(nn.Module):


    def __init__(self, input_dim=16):
        super().__init__()
        self.channels = 1
        self.width = 28
        self.height = 28
        self.input_dim = input_dim

        # Layers
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 64 * 11 * 11),
            nn.ConvTranspose2d(64, 32, 4, 2),
            nn.ConvTranspose2d(32, 1, 4)
        )


    def forward(self, x):
        return self.layers(x)
