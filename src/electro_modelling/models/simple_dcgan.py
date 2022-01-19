# -*- coding: utf-8 -*-

"""

"""

import torch
import torch.nn as nn

from electro_modelling.models.base import GAN


class SimpleGAN(GAN):
    """

    Parameters
    ----------
    z_dim
    dataset
    img_chan
    """

    def __init__(self, z_dim, dataset="MNIST", img_chan=1, nb_fixed_noise=4, operator=None):
        super().__init__(
            z_dim=z_dim,
            model_name="simple_dcgan",
            init_weights=True,
            dataset=dataset,
            img_chan=img_chan,
            nb_fixed_noise=nb_fixed_noise,
            operator=operator,
        )

    def _init_optimizer(self, learning_rate, beta_1=0.5, beta_2=0.999):
        """

        Parameters
        ----------
        learning_rate
        beta_1
        beta_2

        Returns
        -------

        """
        self.gen_opt = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
        )
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
        )

    def _init_criterion(self):
        """

        Returns
        -------

        """
        self.criterion = nn.BCEWithLogitsLoss()

    def _compute_disc_loss(self, real, fake, disc_real_pred, disc_fake_pred):
        """

        Parameters
        ----------
        real
        fake
        disc_real_pred
        disc_fake_pred

        Returns
        -------

        """
        disc_fake_loss = self.criterion(
            disc_fake_pred, torch.zeros_like(disc_fake_pred)
        )
        disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        # compute the global discriminator loss as the mean between fake and real batches losses
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        losses = [disc_loss.item(), disc_fake_loss.item(), disc_real_loss.item()]
        losses_names = ["Total loss", "Fake prediction loss", "Real prediction loss"]
        return disc_loss, losses, losses_names

    def _compute_gen_loss(self, disc_fake_pred):
        """

        Parameters
        ----------
        disc_fake_pred

        Returns
        -------

        """
        gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss
