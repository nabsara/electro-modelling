# -*- coding: utf-8 -*-

"""

"""

import torch.nn as nn

from electro_modelling.models.networks.base import GNet


class DCGANGenerator(GNet):
    """
    Generator Class

    Parameters
    ----------
    z_dim : int
        the dimension of the noise vector
    img_chan : int
        the number of channels in the images ex : 3 for RGB
        (MNIST is black-and-white, so default is 1 channel)
    hidden_dim : int
        the inner dimension

    Attributes
    ----------
        z_dim : int
            the dimension of the noise vector
        model : nn.Sequential
            the GAN generator model
    """

    def __init__(self, z_dim, img_chan=1, hidden_dim=64):
        super().__init__(z_dim=z_dim, img_chan=img_chan, hidden_dim=hidden_dim)
        self.model = self._build_network()

    def _build_network(self):
        """

        Returns
        -------

        """
        return nn.Sequential(
            self._make_gen_block(
                input_channels=self.z_dim,
                output_channels=self.hidden_dim * 4,
                kernel_size=3,
                stride=2,
            ),
            self._make_gen_block(
                input_channels=self.hidden_dim * 4,
                output_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=1,
            ),
            self._make_gen_block(
                input_channels=self.hidden_dim * 2,
                output_channels=self.hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ConvTranspose2d(self.hidden_dim, self.img_chan, kernel_size=4, stride=2),
            nn.Tanh(),
        )

    def _make_gen_block(self, input_channels, output_channels, kernel_size, stride):
        """
        Build the sequence of operations corresponding to a generator
        block of GAN :
        - transposed convolution
        - batchnorm
        - ReLU activation

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
            nn.ConvTranspose2d(
                input_channels, output_channels, kernel_size, stride=stride
            ),
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU(inplace=True),
        )
