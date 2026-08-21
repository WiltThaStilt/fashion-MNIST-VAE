from torch import nn
import torch.nn.functional as F
import torch

class Decoder(nn.Module):


    def __init__(self, input_dim=16):
        super().__init__()
        self.channels = 1
        self.width = 28
        self.height = 28
        self.input_dim = input_dim

        # Layers
        self.input_hidden_layer = nn.Sequential(
            nn.Linear(self.input_dim, 8 * 6 * 6),
            nn.LeakyReLU(0.2),
            nn.Linear(8 * 6 * 6, 16 * 11 * 11),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (16, 11, 11))
        )

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 4, 2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        self.skip_projection = nn.ConvTranspose2d(16, 32, 4, 2, output_padding=1)
        self.output_conv = nn.ConvTranspose2d(32, 1, 4)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))


    def forward(self, x):
        unflattened_x = self.input_hidden_layer(x)
        first_conv_output = self.layers(unflattened_x)

        residual_x = self.skip_projection(unflattened_x)
        second_conv_input = first_conv_output + self.residual_weight * residual_x

        return self.output_conv(second_conv_input)
