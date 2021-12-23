import torch
import torch.nn as nn

from electro_modelling.models.dcgan import DCGAN


class LeastSquareGAN(DCGAN):

    def __init__(self, z_dim):
        super().__init__(
            z_dim=z_dim,
            model_name="least_square_dcgan",
            init_weights=True
        )

    def _init_optimizer(self, learning_rate, beta_1 = 0.5, beta_2 = 0.999):
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

#     def _init_criterion(self):
#         self.criterion = nn.BCELoss()

    def _compute_disc_loss(self, real, fake, disc_real_pred, disc_fake_pred):
        disc_fake_loss = 0.5 * torch.mean((disc_fake_pred-torch.ones_like(disc_fake_pred))**2)
        disc_real_loss = 0.5 * torch.mean((disc_real_pred-torch.zeros_like(disc_real_pred))**2)
        disc_loss = (disc_fake_loss + disc_real_loss) /2
        return disc_loss


    def _compute_gen_loss(self, disc_real_pred, disc_fake_pred):
        gen_loss = 0.5 * torch.mean((disc_fake_pred-torch.ones_like(disc_real_pred))**2)
        return gen_loss
