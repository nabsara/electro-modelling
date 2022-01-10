import torch
import torch.nn as nn

from electro_modelling.models.networks.base import GNet


class GANSynthGenerator(GNet):

    def __init__(self, z_dim, img_chan, hidden_dim, init_kernel=(2, 2)):
        super().__init__(
            z_dim=z_dim,
            img_chan=img_chan,
            hidden_dim=hidden_dim
        )
        self.init_kernel = init_kernel
        self.model = self._build_network()

    def _build_network(self) -> nn.Sequential:
        if self.init_kernel == (16, 2) or self.init_kernel == (2, 2):
            init_block = nn.Sequential(
                # block 1: (1, 1, 256) --> (4, 32, 256)
                self._make_gen_block(
                    input_channels=self.z_dim,
                    output_channels=self.hidden_dim * 8,
                    first=True
                ),

                # block 2 : (4, 32, 256) --> (8, 64, 256)
                self._make_gen_block(
                    input_channels=self.hidden_dim * 8,
                    output_channels=self.hidden_dim * 8,
                ),

                # block 3 : (8, 64, 256) --> (16, 128, 256)
                self._make_gen_block(
                    input_channels=self.hidden_dim * 8,
                    output_channels=self.hidden_dim * 8,
                ),

                # block 4 : (16, 128, 256) --> (32, 256, 256)
                self._make_gen_block(
                    input_channels=self.hidden_dim * 8,
                    output_channels=self.hidden_dim * 8,
                ),
            )
        elif self.init_kernel == (16, 16):
            init_block = nn.Sequential(
                # block 1: (1, 1, 256) --> (32, 32, 256)
                self._make_gen_block(
                    input_channels=self.z_dim,
                    output_channels=self.hidden_dim * 8,
                    first=True
                ),
            )
        else:
            raise NotImplementedError
        return nn.Sequential(
            init_block,

            # block 5 : (32, 256, 256) --> (64, 512, 128)
            self._make_gen_block(
                input_channels=self.hidden_dim * 8,
                output_channels=self.hidden_dim * 4,
            ),

            # block 6 : (64, 512, 128) --> (128, 1024, 64)
            self._make_gen_block(
                input_channels=self.hidden_dim * 4,
                output_channels=self.hidden_dim * 2,
            ),

            # Final layer: (128, 1024, 64) --> (128, 1024, 32)
            self._make_gen_block(
                input_channels=self.hidden_dim * 2,
                output_channels=self.hidden_dim,
                final=True
            ),
            # (128, 1024, 32) --> (128, 1024, 2)
            nn.Conv2d(self.hidden_dim, self.img_chan, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh(),
        )

    def _make_gen_block(self, input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), first=False, final=False):
        if first:

            return nn.Sequential(
                # nn.ConvTranspose2d(input_channels, output_channels, (2 * self.nmel_ratio, 2), stride=stride),
                # TODO : change first kernel size depending on the expected output shape
                # ex: (16, 2) --> (1024, 128)
                # ex : (2, 2) --> (128, 128)
                # ex : (16, 16) --> (128, 128)
                nn.ConvTranspose2d(input_channels, output_channels, self.init_kernel, stride=stride),
                PixelNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size, stride=stride, padding="same"),
                PixelNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Upsample(scale_factor=2),
            )
        elif final:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding="same"),
                PixelNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size, stride=stride, padding="same"),
                PixelNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding="same"),
                PixelNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size, stride=stride, padding="same"),
                PixelNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Upsample(scale_factor=2),
            )


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
