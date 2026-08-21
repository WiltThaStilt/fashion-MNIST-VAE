from torch import nn
import torch

class Encoder(nn.Module):


    def __init__(self, output_dim=16):
        super().__init__()
        self.channels = 1
        self.width = 28
        self.height = 28
        self.output_dim = output_dim

        # Layers
        self.conv_first = nn.Conv2d(in_channels=self.channels, out_channels=32, kernel_size=4) # [(28 - 4) / 1] + 1 = 25 => 32 x 25 x 25

        self.post_first_conv = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2), # [(25 - 4) / 2] + 1 = 11 => 64 x 11 x 11
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.flatten = nn.Flatten()
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        # Parallel Mean and LogVar Layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(16 * 11 * 11 + 32 * 25 * 25, int((16 * 11 * 11 + 32 * 25 * 25) / 3)),
            nn.LeakyReLU(0.2),
            nn.Linear(int((16 * 11 * 11 + 32 * 25 * 25) / 3), int((16 * 11 * 11 + 32 * 25 * 25) / 5)),
            nn.LeakyReLU(0.2)
        )

        self.mu = nn.Linear(int((16 * 11 * 11 + 32 * 25 * 25) / 5), self.output_dim)
        self.logvar = nn.Linear(int((16 * 11 * 11 + 32 * 25 * 25) / 5), self.output_dim)


    def forward(self, x):
        first_res = self.conv_first.forward(x)
        post_first_res = self.post_first_conv(first_res)
        flattened_first_res = self.flatten(first_res)
        flattened_post_first_res = self.flatten (post_first_res)

        combined = torch.cat((self.residual_weight * flattened_first_res, flattened_post_first_res), dim=1)
        hidden_output = self.hidden_layer(combined)

        return self.mu(hidden_output), self.logvar(hidden_output)
