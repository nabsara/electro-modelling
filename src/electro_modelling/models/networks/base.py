# -*- coding: utf-8 -*-
"""

"""

import torch.nn as nn


class GNet(nn.Module):
    """

    Attributes
    ----------

    Parameters
    ----------
    z_dim
    img_chan
    hidden_dim
    """

    def __init__(self, z_dim, img_chan, hidden_dim):
        super().__init__()
        self.z_dim = z_dim
        self.img_chan = img_chan
        self.hidden_dim = hidden_dim

    def _build_network(self) -> nn.Sequential:
        """

        Returns
        -------
            nn.Sequential instance
        """
        raise NotImplementedError

    def _make_gen_block(
        self, input_channels, output_channels, kernel_size, stride, **kwargs
    ):
        """

        Parameters
        ----------
        input_channels
        output_channels
        kernel_size
        stride
        kwargs

        Returns
        -------
            nn.Sequential instance
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Function that applies the forward pass of the generator model
        on a given noise tensor and which returns generated images.

        Parameters
        ----------
        x : Tensor
            a noise tensor with dimensions (n_samples, z_dim)

        Returns
        -------
            generated images
        """
        noise = x.view(len(x), self.z_dim, 1, 1)
        output = self.model(noise)
        # output = output.view(x.size(0), 1, 28, 28)
        return output


class DNet(nn.Module):
    """

    Attributes
    ----------
    img_chan
    hidden_dim
    """

    def __init__(self, img_chan, hidden_dim):
        super().__init__()
        self.img_chan = img_chan
        self.hidden_dim = hidden_dim

    def _build_network(self) -> nn.Sequential:
        """

        Returns
        -------
            nn.Sequential instance
        """
        raise NotImplementedError

    def _make_disc_block(self, input_channels, output_channels, kernel_size, stride):
        """

        Parameters
        ----------
        input_channels
        output_channels
        kernel_size
        stride

        Returns
        -------
            nn.Sequential instance
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Function that applies the forward pass of the discriminator model
        on a given image tensor and which returns a 1-dimensional tensor
        representing fake/real.

        Parameters
        ----------
        x : Tensor
            a flattened image tensor with dimension (img_dim)

        Returns
        -------
            1-dimensional tensor representing fake/real
        """
        predictions = self.model(x)
        output = predictions.view(len(predictions), -1)
        return output
