import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid

from electro_modelling.models.discriminator import Discriminator
from electro_modelling.models.generator import Generator
from electro_modelling.config import settings
from electro_modelling.helpers.helpers_visualization import show_tensor_images


class DCGAN:

    def __init__(self, z_dim, init_weights=True):
        self.z_dim = z_dim
        self.generator = Generator(self.z_dim, img_chan=1, hidden_dim=64).to(device=settings.device)
        self.discriminator = Discriminator(img_chan=1, hidden_dim=16).to(device=settings.device)

        if init_weights:
            self.generator.apply(self.initialize_weights)
            self.discriminator.apply(self.initialize_weights)

        # TODO: add fixed noise for model evaluation and to add to tensorboard
        self.fixed_noise = self.get_noise(64)

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
        Initialize the model weights to the normal distribution
        with mean 0 and standard deviation 0.02

        Parameters
        ----------
        m : nn.Module
            is instance of nn.Conv2d or nn.ConvTranspose2d or nn.BatchNorm2d
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def _init_optimizer(self, learning_rate):
        # TODO: Add RMS Prop for W-GAN
        beta_1 = 0.5
        beta_2 = 0.999
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

    def _init_criterion(self, loss):
        # TODO: Add LeastSquare loss
        # TODO: Add Hinge loss
        # TODO: Add Wasserstein with Gradient penalty loss
        if loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Loss {loss} not implemented in DCGAN")

    def _get_gradient(self, real, fake, epsilon):
        """
        Return the gradient of the critic's scores with respect to mixes of real
        and fake images.

        Parameters
        ----------
        real :
            current batch of real images
        fake :
            current batch of fake images
        epsilon : np.array
            vector of the uniformly random proportions of real / fake per mixed image
        """
        # Mix the images together
        mixed_images = real * epsilon + fake * (1 - epsilon)
        # Calculate the critic's scores on the mixed images
        mixed_scores = self.discriminator(mixed_images)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient

    def _gradient_penalty(self, gradient):
        """
        Return the gradient penalty given a gradient. Compute the magnitude of
        each image's gradient for current batch gradients and penalize the mean
        quadratic distance of each magnitude to 1.

        Parameters
        ----------
        gradient : np.array
            the gradient of the critic's scores with respect to the mixed image

        Returns
        -------
            the gradient penalty
        """
        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)
        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)
        # Penalize the mean squared distance of the gradient norms from 1
        penalty = torch.mean((gradient_norm - 1) * (gradient_norm - 1))
        return penalty

    def get_crit_loss(self, crit_fake_pred, crit_real_pred, gp, c_lambda):
        """
        Return the loss of a critic given the critic's scores for fake and real images,
        the gradient penalty, and gradient penalty weight.
        Parameters
        ----------
            crit_fake_pred: the critic's scores of the fake images
            crit_real_pred: the critic's scores of the real images
            gp: the unweighted gradient penalty
            c_lambda: the current weight of the gradient penalty

        Returns
        -------
            crit_loss: a scalar for the critic's loss, accounting for the relevant factors
        """
        crit_loss = torch.mean(crit_fake_pred - crit_real_pred + c_lambda * gp)
        return crit_loss

    def get_gen_loss(self, crit_fake_pred):
        """
        Return the loss of a generator given the critic's scores of the generator's fake images.

        Parameters
        ----------
        crit_fake_pred :
            the critic's scores of the fake images

        Returns
        -------
            gen_loss: a scalar loss value for the current batch of the generator
        """
        gen_loss = - torch.mean(crit_fake_pred)
        return gen_loss

    def train(self, train_dataloader, loss="bce", lr=0.0002, k_update_only_disc=1, n_epochs=50, display_step=500, models_dir=settings.MODELS_DIR):
        # TODO: Add save model checkpoints and resume training from checkpoints
        start = time.time()

        # define generator and discriminator loss and optimizers
        self._init_optimizer(lr)
        self._init_criterion(loss)

        d_losses = torch.zeros(n_epochs)
        g_losses = torch.zeros(n_epochs)
        img_list = []
        it = 0
        for epoch in range(n_epochs):
            cur_step = 0
            g_loss = 0
            d_loss = 0
            for real, _ in tqdm(train_dataloader):
                cur_batch_size = len(real)
                real = real.to(settings.device)

                mean_disc_loss = 0
                # train discriminator for k steps:
                for _ in range(k_update_only_disc):
                    self.disc_opt.zero_grad()
                    # generate fake data from latent vectors
                    fake_noise = self.get_noise(cur_batch_size)
                    fake = self.generator(fake_noise)
                    # compute discriminator loss on fake and real data
                    disc_fake_pred = self.discriminator(fake.detach())
                    disc_real_pred = self.discriminator(real)
                    if loss == "w-gp":
                        epsilon = torch.rand(len(real), 1, 1, 1, device=settings.device, requires_grad=True)
                        gradient = self._get_gradient(real, fake.detach(), epsilon)
                        gp = self._gradient_penalty(gradient)
                        c_lambda = 10
                        disc_loss = self.get_crit_loss(disc_fake_pred, disc_real_pred, gp, c_lambda)
                    else:
                        disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                        disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                        # compute the global discriminator loss as the mean between fake and real batches losses
                        disc_loss = (disc_fake_loss + disc_real_loss) / 2
                    mean_disc_loss += disc_loss.item() / k_update_only_disc
                    # update discriminator gradients
                    disc_loss.backward(retain_graph=True)
                    # update discriminator optimizer
                    self.disc_opt.step()
                    # keep track of the discriminator loss
                d_loss += mean_disc_loss

                # train generator:
                self.gen_opt.zero_grad()
                # generate fake data from latent vectors
                fake_noise_2 = self.get_noise(cur_batch_size)
                fake_2 = self.generator(fake_noise_2)
                # compute the generator loss on fake data
                disc_fake_pred_2 = self.discriminator(fake_2)
                if loss == "w-gp":
                    gen_loss = self.get_gen_loss(disc_fake_pred_2)
                else:
                    gen_loss = self.criterion(disc_fake_pred_2, torch.ones_like(disc_fake_pred_2))
                # update generator gradients
                gen_loss.backward()
                # update generator optimizer
                self.gen_opt.step()
                # keep track of the average generator loss
                g_loss += gen_loss.item()

                # plot training stats
                if it % 100 == 0 and it > 0:
                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{cur_step}/{len(train_dataloader)}]"
                        f"\tTime: {time.time() - start} (s)\tG_loss: {gen_loss.item()}\tD_loss: {mean_disc_loss.item()}"
                        f"\tD(x): {disc_real_pred.mean().item()}"
                        f"\tD(G(z)): {disc_fake_pred.mean().item()} / {disc_fake_pred_2.mean().item()}")
                cur_step += 1

                # Check how the generator is doing by saving G's output on fixed_noise
                if it % display_step == 0 or ((epoch == n_epochs - 1) and (cur_step == len(train_dataloader) - 1)):
                    # TODO: Add to tensorboard
                    show_tensor_images(fake)
                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu()
                    img_list.append(make_grid(fake, padding=2, normalize=True))
                it += 1
            # keep track on batch mean losses evolution through epochs
            d_losses[epoch] = d_loss / len(train_dataloader)
            g_losses[epoch] = g_loss / len(train_dataloader)

            # model checkpoints:
            if (epoch % 10 == 0 or epoch == n_epochs - 1) and epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.gen_opt.state_dict(),
                    'loss': g_losses[epoch],
                }, os.path.join(
                    models_dir,
                    f'generator_dcgan_mnist_{loss}_{lr}_{k_update_only_disc}_{n_epochs}.pt')
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.disc_opt.state_dict(),
                    'loss': d_losses[epoch],
                }, os.path.join(
                    models_dir,
                    f'discriminator_dcgan_mnist_{loss}_{lr}_{k_update_only_disc}_{n_epochs}.pt')
                )

        return d_losses, g_losses, img_list

    def evaluate(self):
        pass

    def save_models(self, generator_filename='generator_dcgan_mnist.pt', discriminator_filename="discriminator_dcgan_mnist.pt"):
        torch.save(self.generator.state_dict(), os.path.join(settings.MODELS_DIR, generator_filename))
        torch.save(self.discriminator.state_dict(), os.path.join(settings.MODELS_DIR, discriminator_filename))




