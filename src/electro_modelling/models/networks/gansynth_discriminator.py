import torch
import torch.nn as nn

from electro_modelling.models.networks.base import DNet


class GANSynthDiscriminator(DNet):
    """
        Discriminator Class

        Parameters
        ----------
        img_chan : int
            the number of channels in the images ex : 3 for RGB
            (MNIST is black-and-white, so default is 1 channel)
        hidden_dim : int
            the inner dimension

        Attributes
        ----------
            model : nn.Sequential
                the GAN discriminator model
        """

    def __init__(self, img_chan, hidden_dim=32, init_kernel=(2, 2)):
        super().__init__(img_chan=img_chan, hidden_dim=hidden_dim)
        self.init_kernel = init_kernel
        self.model = self._build_network()

    def _build_network(self) -> nn.Sequential:
        if True:#self.init_kernel == (2, 2) or self.init_kernel == (16, 2):
            final_block = nn.Sequential(
                # block 4: (16, 128, 128) --> (8, 64, 256)
                self._make_disc_block(
                    input_channels=self.hidden_dim * 4,
                    output_channels=self.hidden_dim * 8,
                ),

                # block 5: (8, 64, 256) --> (4, 32, 256)
                self._make_disc_block(
                    input_channels=self.hidden_dim * 8,
                    output_channels=self.hidden_dim * 8,
                ),

                # block 6: (4, 32, 256) --> (2, 16, 256)
                self._make_disc_block(
                    input_channels=self.hidden_dim * 8,
                    output_channels=self.hidden_dim * 8,
                ),

                # Final layer:
                self._make_disc_block(
                    input_channels=self.hidden_dim * 8,
                    output_channels=self.hidden_dim * 8,
                    final=True
                ),
                nn.Conv2d(self.hidden_dim * 8, 1, kernel_size=self.init_kernel, stride=(1, 1)),
            )
        elif self.init_kernel == (16, 16):
            final_block = nn.Sequential(
                # Final layer:
                self._make_disc_block(
                    input_channels=self.hidden_dim * 4,
                    output_channels=self.hidden_dim * 4,
                    final=True
                ),
                nn.Conv2d(self.hidden_dim * 4, 1, kernel_size=(16, 16), stride=(1, 1)),
            )
        else:
            raise NotImplementedError

        return nn.Sequential(
            nn.Conv2d(self.img_chan, self.hidden_dim, kernel_size=(1, 1), stride=(1, 1)),

            # block 1 : (128, 1024, 32) --> (64, 512, 32)
            self._make_disc_block(
                input_channels=self.hidden_dim,
                output_channels=self.hidden_dim,
            ),

            # block 2: (64, 512, 32) --> (32, 256, 64)
            self._make_disc_block(
                input_channels=self.hidden_dim,
                output_channels=self.hidden_dim * 2,
            ),

            # block 3: (32, 256, 64) --> (16, 128, 128)
            self._make_disc_block(
                input_channels=self.hidden_dim * 2,
                output_channels=self.hidden_dim * 4,
            ),

            final_block
        )

    def _make_disc_block(
            self, input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), final=False
    ):
        """
        Build the sequence of operations corresponding to a discriminator
        block of GAN :
        - convolution
        - batchnorm
        - LeakyReLU activation

        Parameters
        ----------
        input_channels : int
            the number of channels of the input feature representation
        output_channels : int
            the number of channels of the output feature representation
        kernel_size : tuple
            the size of each convolutional filter (kernel_size, kernel_size)
        stride : tuple
            the stride of the convolution

        Returns
        -------
            nn.Sequential GAN generator block
        """
        if final:
            return nn.Sequential(
                # minibatch std : (2, 16, 256) --> (2, 16, 257)
                MinibatchStd(),
                # (2, 16, 257) --> (2, 16, 256)
                nn.Conv2d(input_channels + 1, output_channels, kernel_size, stride=stride, padding="same"),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # (2, 16, 256) --> (2, 16, 256)
                nn.Conv2d(output_channels, output_channels, kernel_size, stride=stride, padding="same"),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding="same"),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size, stride=stride, padding="same"),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )


class MinibatchStd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)
