import os
import time
import matplotlib.pyplot as plt

from electro_modelling.helpers.utils import save_pickle
from electro_modelling.datasets.mnist_data_loader import mnist_data_loader
from electro_modelling.models.dcgan import DCGAN


class MNISTPipeline:

    def __init__(self, data_dir, models_dir, batch_size, z_dims):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.batch_size = batch_size
        self.z_dim = z_dims
        self.train_loader, self.test_loader = mnist_data_loader(self.batch_size, data_dir=self.data_dir)
        self.model = DCGAN(z_dim=self.z_dim, init_weights=True)

    def train(self, loss, learning_rate, k_disc_steps, n_epochs, display_step):
        d_loss, g_loss, img_list = self.model.train(
            train_dataloader=self.train_loader,
            loss=loss,
            lr=learning_rate,
            k_update_only_disc=k_disc_steps,
            n_epochs=n_epochs,
            display_step=display_step,
            models_dir=self.models_dir
        )
        results = {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "img_list": img_list
        }
        save_pickle(
            results,
            os.path.join(
                self.models_dir,
                f"results_loss_dcgan_mnist_{loss}_{learning_rate}_{k_disc_steps}_{n_epochs}.pkl"
            )
        )

    # TODO: to remove
    @staticmethod
    def plot_gan_losses(d_loss, g_loss):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(d_loss.detach(), label="discriminator loss")
        ax.plot(g_loss.detach(), label="generator loss")
        ax.set_xlabel("n_epochs")
        ax.legend()
        ax.grid(True)
        plt.show()
