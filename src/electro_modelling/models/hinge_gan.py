import torch
import torch.nn as nn

from electro_modelling.models.dcgan import DCGAN


class MyHingeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, output, target):
        return (torch.mean(torch.maximum(1 - torch.mul(output, target),0*target),0)) 
    
class MyLinearLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.mean(torch.mul(target,output),0)


class HingeGAN(DCGAN):

    def __init__(self, z_dim):
        super().__init__(
            z_dim=z_dim,
            model_name="hinge_dcgan",
            init_weights=True
        )

    def _init_optimizer(self, learning_rate, beta_1 = 0.5, beta_2 = 0.999):
        self.gen_opt = torch.optim.Adam(self.generator.parameters(),
lr=learning_rate, betas=(beta_1, beta_2))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(),
lr=learning_rate, betas=(beta_1, beta_2))

    def _init_criterion1(self):
        self.criterion1 = MyHingeLoss()
        
    def _init_criterion2(self):
        self.criterion2 = MyLinearLoss()

    def _compute_disc_loss(self, real, fake, disc_real_pred, disc_fake_pred):
        disc_fake_loss = self.criterion1(disc_fake_pred,
torch.zeros_like(disc_fake_pred))
        disc_real_loss = self.criterion1(disc_real_pred,
torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) /2
        return disc_loss

    def _compute_gen_loss(self, disc_fake_pred):
        gen_loss = self.criterion2(disc_fake_pred,
torch.ones_like(disc_fake_pred))
        return gen_loss
