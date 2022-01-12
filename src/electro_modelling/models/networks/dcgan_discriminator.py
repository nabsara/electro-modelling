# -*- coding: utf-8 -*-
"""

"""

import torch.nn as nn

from electro_modelling.models.networks.base import DNet


class DCGANDiscriminator(DNet):
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

    def __init__(self, img_chan=1, hidden_dim=16):
        super().__init__(img_chan=img_chan, hidden_dim=hidden_dim)
        self.model = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """

        Returns
        -------

        """
        return nn.Sequential(
            self._make_disc_block(
                input_channels=self.img_chan,
                output_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
            ),
            self._make_disc_block(
                input_channels=self.hidden_dim,
                output_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
            ),
            nn.Conv2d(self.hidden_dim * 2, 1, kernel_size=4, stride=2),
        )

    def _make_disc_block(
        self, input_channels, output_channels, kernel_size=4, stride=2
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
        kernel_size : int
            the size of each convolutional filter (kernel_size, kernel_size)
        stride : int
            the stride of the convolution

        Returns
        -------
            nn.Sequential GAN generator block
        """
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride),
            nn.BatchNorm2d(num_features=output_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
