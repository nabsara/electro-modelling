import torch
import torch.nn as nn

from electro_modelling.models.base import GAN


class LeastSquareGAN(GAN):
    def __init__(self, z_dim):
        super().__init__(
            z_dim=z_dim, model_name="least_square_dcgan", init_weights=True
        )

    def _init_optimizer(self, learning_rate, beta_1=0.5, beta_2=0.999):
        self.gen_opt = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
        )
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
        )

    def _init_criterion(self):
        self.criterion = nn.MSELoss()

    def _compute_disc_loss(self, real, fake, disc_real_pred, disc_fake_pred):
        """
        Return the loss of a critic given the critic's scores for fake and
        real images.

        Parameters
        ----------
        real :
            current batch of real images
        fake :
            current batch of fake images
        disc_real_pred :
            the critic's scores of the real images
        disc_fake_pred :
            the critic's scores of the fake images

        Returns
        -------
            a scalar for the critic's loss for the current batch
        """
        disc_fake_loss = self.criterion(
            disc_fake_pred, torch.zeros_like(disc_fake_pred)
        )
        disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss

    def _compute_gen_loss(self, disc_fake_pred):
        """
        Return the loss of a generator given the critic's scores of the
        generator's fake images.

        Parameters
        ----------
        disc_fake_pred :
            the critic's scores of the fake images

        Returns
        -------
            a scalar for the generator loss for the current batch
        """
        gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss
