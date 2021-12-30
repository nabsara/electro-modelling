import os

from electro_modelling.helpers.utils import save_pickle
from electro_modelling.datasets.mnist_data_loader import mnist_data_loader
from electro_modelling.models.models import models


class MNISTPipeline:

    def __init__(self, model_name, data_dir, models_dir, batch_size, z_dims):
        if model_name not in list(models.keys()):
            raise ValueError(f"Model named {model_name} not implemented. Try models in : {list(models.keys())}")

        self.data_dir = data_dir
        self.models_dir = models_dir
        self.batch_size = batch_size
        self.z_dim = z_dims
        self.train_loader, self.test_loader = mnist_data_loader(self.batch_size, data_dir=self.data_dir)
        self.model = models[model_name](z_dim=self.z_dim)

    def train(self, learning_rate, k_disc_steps, n_epochs, display_step, show_fig=False):
        d_loss, g_loss, img_list = self.model.train(
            train_dataloader=self.train_loader,
            lr=learning_rate,
            k_disc_steps=k_disc_steps,
            n_epochs=n_epochs,
            display_step=display_step,
            models_dir=self.models_dir,
            show_fig=show_fig
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
                f"results_loss_mnist__{self.model.model_name}__z_{self.z_dim}__lr_{learning_rate}__k_{k_disc_steps}__e_{n_epochs}.pkl"
            )
        )
