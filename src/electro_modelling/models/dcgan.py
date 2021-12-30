import os
import time
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import dataset
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from electro_modelling.models.discriminator import Discriminator
from electro_modelling.models.generator import Generator
from electro_modelling.config import settings
from electro_modelling.helpers.helpers_visualization import show_tensor_images
from electro_modelling.helpers.helpers_audio import plot_spectrogram_mag,image_grid_spectrograms
from electro_modelling.datasets.signal_processing import SignalOperators


class DCGAN:

    def __init__(self,z_dim, model_name, init_weights=True,dataset='MNIST',img_chan=1,operator=None):
        self.z_dim = z_dim
        self.dataset = dataset
        self.operator = operator
        self.nmel_ratio = int(operator.nmels/operator.nb_trames)
        self.generator = Generator(dataset,self.z_dim, img_chan, hidden_dim=32,nmel_ratio=self.nmel_ratio).to(device=settings.device)
        self.discriminator = Discriminator(dataset,img_chan, hidden_dim=32,nmel_ratio=self.nmel_ratio).to(device=settings.device)

        self.model_name = model_name
        if self.model_name == 'wgan':
            self.nb_loss_disc = 5
        else:
            self.nb_loss_disc = 3
        self.gen_opt = None
        self.disc_opt = None
        
        

        if init_weights:
            self.generator.apply(self.initialize_weights)
            self.discriminator.apply(self.initialize_weights)

        # TODO: add fixed noise for model evaluation and to add to tensorboard
        self.fixed_noise = self.get_noise(4)

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
        noise = torch.randn(n_samples, self.z_dim, device=settings.device)
        norm = torch.norm(noise,dim=1,keepdim=True)
        return noise/norm
    
        
    def get_sounds(self,fakes):
        sounds_list = []
        for i,fake in enumerate(fakes):
            STFT_mel_amp = fake[0].numpy()
            sound = self.operator.backward(STFT_mel_amp)
            sounds_list.append(torch.tensor(sound))
        sounds_tensor=torch.stack(sounds_list)
        return (sounds_tensor)

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

    def train(self, train_dataloader, lr=0.0002, k_disc_steps=1, n_epochs=50, display_step=500, models_dir=settings.MODELS_DIR, show_fig=False):
        # TODO: Add save model checkpoints and resume training from checkpoints
        start = time.time()
        # defining a SummaryWriter to write information to TensorBoard during training
        writer = SummaryWriter(os.path.join(
            models_dir, f"runs/exp__{self.model_name}__z_{self.z_dim}__lr_{lr}__k_{k_disc_steps}__e_{n_epochs}_"+time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()))
        )

        # define generator and discriminator loss and optimizers
        self._init_optimizer(lr)
        self._init_criterion()

        d_losses = torch.zeros(n_epochs)
        g_losses = torch.zeros(n_epochs)
        img_list = []
        it = 0
        for epoch in range(n_epochs):
            cur_step = 0
            g_loss = 0
            d_loss = 0
            d_losses = np.zeros(self.nb_loss_disc)
            d_display_losses = np.zeros(self.nb_loss_disc)
            g_display_loss = 0
            for real in tqdm(train_dataloader):
                if self.dataset=='MNIST':
                    _,real = real
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
                    disc_loss,losses,losses_names = self._compute_disc_loss(real, fake, disc_real_pred, disc_fake_pred)
                    mean_disc_losses += np.array(losses)/k_disc_steps
                    # update discriminator gradients
                    disc_loss.backward(retain_graph=True)
                    # update discriminator optimizer
                    self.disc_opt.step()
                # keep track of the discriminator loss
                d_losses += mean_disc_losses
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
                if it % display_step == 0 or ((epoch == n_epochs - 1) and (cur_step == len(train_dataloader) - 1)):
                    
                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu()
                        if show_fig:
                            if self.dataset=='techno':
                                imgs = fake
                                figure = image_grid_spectrograms(imgs)
                                figure.show()
                            else:
                                show_tensor_images(fake)
                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{cur_step}/{len(train_dataloader)}]"
                        f"\tTime: {time.time() - start} (s)\tG_loss: {g_display_loss / display_step}\tTotal_D_loss: {d_display_losses[0] / display_step}"
                    )
                    
                    # Add training losses and fake images evolution to tensorboard
                    writer.add_scalar(
                        "training generator loss",
                        g_display_loss / display_step,
                        epoch * len(train_dataloader) + cur_step
                    )
                    for loss,name in zip(d_display_losses,losses_names):
                        writer.add_scalar(
                            'Discriminator Losses/'+name,
                            loss / display_step,
                            epoch * len(train_dataloader) + cur_step
                        )
                    if self.dataset == 'MNIST':
                        writer.add_image(
                            "generated_images",
                            make_grid(fake),
                            epoch * len(train_dataloader) + cur_step
                        )
                    if self.dataset=='techno':
                        imgs = fake
                        sounds_tensor = self.get_sounds(imgs)
                        figure = image_grid_spectrograms(imgs)
                        
                        writer.add_figure('generated_images',figure, epoch * len(train_dataloader) + cur_step)
                        for j in range(sounds_tensor.shape[0]):
                            writer.add_audio('generated_sound/'+str(j), sounds_tensor[j],global_step = epoch * len(train_dataloader) + cur_step,sample_rate = 16000)
                            
                    img_list.append(make_grid(fake, padding=2, normalize=True))
                    d_display_losses = np.zeros(self.nb_loss_disc)
                    g_display_loss = 0
                cur_step += 1
                it += 1
            # keep track on batch mean losses evolution through epochs
            d_losses[epoch] = d_loss / len(train_dataloader)
            g_losses[epoch] = g_loss / len(train_dataloader)

            # model checkpoints:
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.gen_opt.state_dict(),
                    'loss': g_losses[epoch],
                }, os.path.join(
                    models_dir,
                    f'generator__{self.model_name}__z_{self.z_dim}__lr_{lr}__k_{k_disc_steps}__e_{n_epochs}.pt')
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.disc_opt.state_dict(),
                    'loss': d_losses[epoch],
                }, os.path.join(
                    models_dir,
                    f'discriminator__{self.model_name}__z_{self.z_dim}__lr_{lr}__k_{k_disc_steps}__e_{n_epochs}.pt')
                )

        return d_losses, g_losses, img_list

    def evaluate(self):
        pass

    def save_models(self, generator_filename='generator_dcgan.pt', discriminator_filename="discriminator_dcgan.pt"):
        torch.save(self.generator.state_dict(), os.path.join(settings.MODELS_DIR, generator_filename))
        torch.save(self.discriminator.state_dict(), os.path.join(settings.MODELS_DIR, discriminator_filename))




