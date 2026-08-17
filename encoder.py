from torch import nn


class Encoder(nn.Module):


    def __init__(self, output_dim=16):
        super().__init__()
        self.channels = 1
        self.width = 28
        self.height = 28
        self.output_dim = output_dim

        # Layers
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=32, kernel_size=4), # [(28 - 4) / 1] + 1 = 25 => 32 x 25 x 25
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), # [(25 - 4) / 2] + 1 = 11 => 64 x 11 x 11
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Flatten()
        )

        # Parallel Mean and LogVar Layer
        self.mu = nn.Linear(64 * 11 * 11, self.output_dim)
        self.logvar = nn.Linear(64 * 11 * 11, self.output_dim)


    def forward(self, x):
        convolutional_fan_out = self.layers(x)
        return self.mu(convolutional_fan_out), self.logvar(convolutional_fan_out)
