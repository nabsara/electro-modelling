# -*- coding: utf-8 -*-
"""

"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from electro_modelling.config import settings
from electro_modelling.helpers.helpers_visualization import show_tensor_images
from electro_modelling.helpers.helpers_audio import image_grid_spectrograms
from electro_modelling.models.networks import (
    DCGANGenerator,
    DCGANDiscriminator,
    GANSynthGenerator,
    GANSynthDiscriminator,
)


class GAN:
    """

    Parameters
    ----------
    z_dim
    model_name
    init_weights
    dataset
    img_chan
    nb_fixed_noise
    operator

    Attributes
    ----------

    """

    def __init__(
        self,
        z_dim,
        model_name,
        init_weights,
        dataset,
        img_chan,
        nb_fixed_noise=4,
        operator=None,
    ):
        self.z_dim = z_dim
        self.dataset = dataset

        self.suffix_model_name = ""
        if operator is not None:
            self.operator = operator
            self.nmel_ratio = int(operator.nmels / operator.nb_trames)
            self.init_kernel = (int(2 * self.nmel_ratio), 2)
            self.generator = GANSynthGenerator(
                z_dim=self.z_dim,
                img_chan=img_chan,
                hidden_dim=32,
                init_kernel=self.init_kernel,
            ).to(device=settings.device)
            self.discriminator = GANSynthDiscriminator(
                img_chan=img_chan, hidden_dim=32, init_kernel=self.init_kernel
            ).to(device=settings.device)
            self.suffix_model_name = (
                "img_size_"
                + str(self.operator.nmels)
                + "_128__init_kernel_"
                + str(self.init_kernel)
                + "_minibatch_std"
            )
        else:
            self.generator = DCGANGenerator(
                z_dim=self.z_dim, img_chan=img_chan, hidden_dim=64
            ).to(device=settings.device)
            self.discriminator = DCGANDiscriminator(
                img_chan=img_chan, hidden_dim=16
            ).to(device=settings.device)
            # self.generator = Generator(
            #     dataset, self.z_dim, img_chan=1, hidden_dim=64
            # ).to(device=settings.device)
            # self.discriminator = Discriminator(dataset, img_chan=1, hidden_dim=16).to(
            #     device=settings.device
            # )

        self.model_name = model_name
        if self.model_name == "wgan":
            self.nb_loss_disc = 5
        else:
            self.nb_loss_disc = 3
        self.gen_opt = None
        self.disc_opt = None

        if init_weights:
            self.generator.apply(self.initialize_weights)
            self.discriminator.apply(self.initialize_weights)

        # TODO: add fixed noise for model evaluation and to add to tensorboard
        self.fixed_noise = self.get_noise(nb_fixed_noise)

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
        # GANSynth : samples a random vector z from a spherical Gaussian
        noise = torch.randn(n_samples, self.z_dim, device=settings.device)
        norm = torch.norm(noise, dim=1, keepdim=True)
        return noise / norm

    def get_sounds(self, fakes):
        sounds_list = []
        for fake in fakes:
            STFT_mel = fake.numpy().copy()
            sound = self.operator.backward(STFT_mel, unnormalize=True)
            sounds_list.append(torch.tensor(sound))
        sounds_tensor = torch.stack(sounds_list)
        return sounds_tensor

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

    def _init_optimizer(self, *args, **kwargs):

        raise NotImplementedError

    def _init_criterion(self, **kwargs):
        pass

    def _compute_disc_loss(self, real, fake, disc_real_pred, disc_fake_pred):
        raise NotImplementedError

    def _compute_gen_loss(self, disc_fake_pred):
        raise NotImplementedError

    def train(
        self,
        train_dataloader,
        lr=0.0002,
        k_disc_steps=1,
        n_epochs=50,
        display_step=500,
        models_dir=settings.MODELS_DIR,
        show_fig=False,
    ):
        """

        Parameters
        ----------
        train_dataloader
        lr
        k_disc_steps
        n_epochs
        display_step
        models_dir
        show_fig

        Returns
        -------

        """
        # TODO: Add save model checkpoints and resume training from checkpoints
        start = time.time()
        # defining a SummaryWriter to write information to TensorBoard
        # during training
        writer = SummaryWriter(
            os.path.join(
                models_dir,
                f"runs/exp__{self.model_name}_{self.suffix_model_name}__z_{self.z_dim}__lr_{lr}__k_{k_disc_steps}__e_{n_epochs}_"
                + time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()),
            )
        )

        # define generator and discriminator loss and optimizers
        self._init_optimizer(lr)
        self._init_criterion()

        # placeholders to save losses and generated data evolution during training
        d_losses = torch.zeros(n_epochs)
        g_losses = torch.zeros(n_epochs)
        img_list = []

        it = 0  # number of batch iterations updated at the end of the dataloader for loop
        for epoch in range(n_epochs):
            it_display = 0
            cur_step = 0
            g_loss = 0
            d_loss = 0
            d_loss_arr = np.zeros(self.nb_loss_disc)
            d_display_losses = np.zeros(self.nb_loss_disc)
            g_display_loss = 0
            for real in tqdm(train_dataloader):

                if self.dataset == "MNIST":
                    real, _ = real
                cur_batch_size = len(real)
                real = real.to(settings.device)
                mean_disc_losses = np.zeros(self.nb_loss_disc)
                # train discriminator for k steps:
                for _ in range(k_disc_steps):
                    self.disc_opt.zero_grad()
                    # generate fake data from latent vectors
                    fake_noise = self.get_noise(cur_batch_size)
                    fake = self.generator(fake_noise)
                    # compute discriminator loss on fake and real data
                    disc_fake_pred = self.discriminator(fake.detach())
                    disc_real_pred = self.discriminator(real)
                    disc_loss, losses, losses_names = self._compute_disc_loss(
                        real, fake, disc_real_pred, disc_fake_pred
                    )
                    mean_disc_losses += np.array(losses) / k_disc_steps
                    # update discriminator gradients
                    disc_loss.backward(retain_graph=True)
                    # update discriminator optimizer
                    self.disc_opt.step()
                # keep track of the discriminator loss
                d_loss_arr += mean_disc_losses
                d_display_losses += mean_disc_losses
                d_loss = d_losses[0]
                # train generator:
                self.gen_opt.zero_grad()
                # generate fake data from latent vectors
                fake_noise_2 = self.get_noise(cur_batch_size)
                fake_2 = self.generator(fake_noise_2)
                # compute the generator loss on fake data
                disc_fake_pred_2 = self.discriminator(fake_2)
                gen_loss = self._compute_gen_loss(disc_fake_pred_2)
                # update generator gradients
                gen_loss.backward()
                # update generator optimizer
                self.gen_opt.step()
                # keep track of the average generator loss
                g_loss += gen_loss.item()
                g_display_loss += gen_loss.item()
                # display training stats
                # Check how the generator is doing by saving G's output on fixed_noise
                it_display += 1
                if it % display_step == 0 or (
                    (epoch == n_epochs - 1) and (cur_step == len(train_dataloader) - 1)
                ):

                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu()
                        if show_fig:
                            if self.dataset == "techno":
                                imgs = fake
                                figure = image_grid_spectrograms(imgs)
                                figure.show()
                            else:
                                show_tensor_images(fake)
                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{cur_step}/{len(train_dataloader)}]"
                        f"\tTime: {time.time() - start} (s)\tG_loss: {g_display_loss / display_step}"
                        f"\tTotal_D_loss: {d_display_losses[0] / it_display}"
                    )

                    # Add training losses and fake images evolution to tensorboard
                    writer.add_scalar(
                        "training generator loss",
                        g_display_loss / it_display,
                        epoch * len(train_dataloader) + cur_step,
                    )
                    for loss, name in zip(d_display_losses, losses_names):
                        writer.add_scalar(
                            "Discriminator Losses/" + name,
                            loss / it_display,
                            epoch * len(train_dataloader) + cur_step,
                        )
                    if self.dataset == "MNIST":
                        writer.add_image(
                            "generated_images",
                            make_grid(fake),
                            epoch * len(train_dataloader) + cur_step,
                        )
                    if self.dataset == "techno":
                        # Add generated samples to tensorboard
                        imgs_fake = fake
                        # denormalize mel spectrograms
                        fake_sounds_tensor = self.get_sounds(imgs_fake)
                        fake_figure = image_grid_spectrograms(imgs_fake)

                        writer.add_figure(
                            "generated_images",
                            fake_figure,
                            epoch * len(train_dataloader) + cur_step,
                        )
                        for j in range(fake_sounds_tensor.shape[0]):
                            writer.add_audio(
                                "generated_sound/" + str(j),
                                fake_sounds_tensor[j],
                                global_step=epoch * len(train_dataloader) + cur_step,
                                sample_rate=16000,
                            )

                        # Add real samples to tensorboard
                        imgs_real = real[: min(len(real), 4), :, :, :].detach().cpu()

                        # denormalize mel spectrograms
                        # v_max = 2.2926
                        # v_min = -6.0
                        # imgs_real = imgs_real * (0.5 * abs(v_max - v_min)) + 0.5 * (v_max + v_min)
                        real_sounds_tensor = self.get_sounds(imgs_real)
                        real_figure = image_grid_spectrograms(imgs_real)
                        print(torch.min(imgs_real))

                        writer.add_figure(
                            "real_images",
                            real_figure,
                            epoch * len(train_dataloader) + cur_step,
                        )
                        for j in range(real_sounds_tensor.shape[0]):
                            writer.add_audio(
                                "real_sound/" + str(j),
                                real_sounds_tensor[j],
                                global_step=epoch * len(train_dataloader) + cur_step,
                                sample_rate=16000,
                            )

                    img_list.append(make_grid(fake, padding=2, normalize=True))
                    d_display_losses = np.zeros(self.nb_loss_disc)
                    g_display_loss = 0
                    it_display = 0
                cur_step += 1
                it += 1
            # keep track on batch mean losses evolution through epochs
            d_losses[epoch] = d_loss / len(train_dataloader)
            g_losses[epoch] = g_loss / len(train_dataloader)

            # model checkpoints:
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                self.save_models(
                    epoch=epoch,
                    gen_losses=g_losses,
                    disc_losses=d_losses,
                    models_dir=models_dir,
                    generator_filename=f"generator__{self.model_name}_{self.suffix_model_name}__z_{self.z_dim}__lr_{lr}"
                    f"__k_{k_disc_steps}__e_{n_epochs}.pt",
                    discriminator_filename=f"discriminator__{self.model_name}_{self.suffix_model_name}__z_{self.z_dim}"
                    f"__lr_{lr}__k_{k_disc_steps}__e_{n_epochs}.pt",
                )

        return d_losses, g_losses, img_list

    def save_models(
        self,
        epoch,
        gen_losses,
        disc_losses,
        models_dir,
        generator_filename,
        discriminator_filename,
    ):
        """

        Parameters
        ----------
        epoch
        gen_losses
        disc_losses
        models_dir
        generator_filename
        discriminator_filename

        Returns
        -------

        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.generator.state_dict(),
                "optimizer_state_dict": self.gen_opt.state_dict(),
                "loss": gen_losses[epoch],
            },
            os.path.join(models_dir, generator_filename),
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.discriminator.state_dict(),
                "optimizer_state_dict": self.disc_opt.state_dict(),
                "loss": disc_losses[epoch],
            },
            os.path.join(models_dir, discriminator_filename),
        )

    def evaluate(self):
        pass
