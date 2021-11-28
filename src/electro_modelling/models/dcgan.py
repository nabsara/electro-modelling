import os
import torch
import torch.nn as nn
from tqdm import tqdm

from electro_modelling.models.discriminator import Discriminator
from electro_modelling.models.generator import Generator
from electro_modelling.config import settings


class DCGAN:

    def __init__(self, z_dim, init_weights=True):
        self.z_dim = z_dim
        self.generator = Generator(self.z_dim, img_chan=1, hidden_dim=64).to(device=settings.device)
        self.discriminator = Discriminator(img_chan=1, hidden_dim=16).to(device=settings.device)
        if init_weights:
            self.generator.apply(self.initialize_weights)
            self.discriminator.apply(self.initialize_weights)

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def get_noise(self, n_samples):
        """
        Create the noise vectors, tensor of shape (n_samples, z_dim)
        filled with random numbers from the normal distribution.

        Parameters
        ----------
        n_samples : int
            the number of samples to generate
        z_dim : int
            the dimension of the noise vector
        device : str
            the device type 'cpu' or 'cuda'

        Returns
        -------
            the noise tensor of shape (n_samples, z_dim)
        """
        return torch.randn(n_samples, self.z_dim, device=settings.device)

    @staticmethod
    def initialize_weights(m):
        """
        Initiliaze the model weights to the normal distribution
        with mean 0 and standard deviation 0.02

        Parameters
        ----------
        m : nn.Module
            is instance of nn.Conv2d or nn.ConvTranspose2d or nn.BatchNorm2d
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def train(self, train_dataloader, batch_size=128, lr=0.0002, k_update_only_disc=1, n_epochs=50, display_step=500):
        gen = self.get_generator()
        disc = self.get_discriminator()
        # the optimizer's momentum parameters
        # https://distill.pub/2017/momentum/
        beta_1 = 0.5
        beta_2 = 0.999
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta_1, beta_2))
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
        criterion = nn.BCEWithLogitsLoss()

        loss_d = torch.zeros(int(n_epochs / 10))
        loss_g = torch.zeros(int(n_epochs / 10))
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        for epoch in range(n_epochs):
            for i, (real, _) in tqdm(enumerate(train_dataloader)):
                if i % k_update_only_disc == 0:
                    cur_batch_size = len(real)
                    real = real.to(settings.device)

                    # Update discriminator
                    disc_opt.zero_grad()
                    fake_noise = self.get_noise(cur_batch_size)
                    fake = gen(fake_noise)
                    disc_fake_pred = disc(fake.detach())
                    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                    disc_real_pred = disc(real)
                    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                    disc_loss = (disc_fake_loss + disc_real_loss) / 2

                    # Keep track of the average discriminator loss
                    mean_discriminator_loss += disc_loss.item() / display_step
                    # Update gradients
                    disc_loss.backward(retain_graph=True)
                    # Update optimizer
                    disc_opt.step()

                    # Update generator
                    gen_opt.zero_grad()
                    fake_noise_2 = self.get_noise(cur_batch_size)
                    fake_2 = gen(fake_noise_2)
                    disc_fake_pred = disc(fake_2)
                    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                    gen_loss.backward()
                    gen_opt.step()

                    # Keep track of the average generator loss
                    mean_generator_loss += gen_loss.item() / display_step
                else:
                    cur_batch_size = len(real)
                    real = real.to(settings.device)

                    # Update discriminator
                    disc_opt.zero_grad()
                    fake_noise = self.get_noise(cur_batch_size)
                    fake = gen(fake_noise)
                    disc_fake_pred = disc(fake.detach())
                    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                    disc_real_pred = disc(real)
                    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                    disc_loss = (disc_fake_loss + disc_real_loss) / 2

                    # Keep track of the average discriminator loss
                    mean_discriminator_loss += disc_loss.item() / display_step
                    # Update gradients
                    disc_loss.backward(retain_graph=True)
                    # Update optimizer
                    disc_opt.step()

                # Visualization code
                if cur_step % display_step == 0 and cur_step > 0:
                    print(
                        f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                    # show_tensor_images(fake)
                    # show_tensor_images(real)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

                # Show loss
                if epoch % 10 == 0 and i == batch_size - 1:
                    loss_d[int(epoch / 10)] = disc_loss.item()
                    loss_g[int(epoch / 10)] = gen_loss.item()
                cur_step += 1
        return loss_d, loss_g

    def evaluate(self):
        pass

    def save_models(self, generator_filename='generator_dcgan_mnist.pt', discriminator_filename="discriminator_dcgan_mnist.pt"):
        torch.save(self.generator.state_dict(), os.path.join(settings.MODELS_DIR, generator_filename))
        torch.save(self.discriminator.state_dict(), os.path.join(settings.MODELS_DIR, discriminator_filename))




