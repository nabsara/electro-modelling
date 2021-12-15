import click
from electro_modelling.config import settings
from electro_modelling.pipelines.mnist_pipeline import MNISTPipeline


@click.option(
    "--data_dir",
    default=settings.DATA_DIR,
    help="Absolute path to data directory",
)
@click.option(
    "--models_dir",
    default=settings.MODELS_DIR,
    help="Absolute path to models directory",
)
@click.option(
    "--batch_size",
    default=128,
    help="Data loader batch size",
)
@click.option(
    "--z_dims",
    default=10,
    help="Dimension of the noise vector (latent space)",
)
@click.option(
    "--loss",
    default="bce",
    help="Name of the training loss",
)
@click.option(
    "--n_epochs",
    default=50,
    help="Number of epochs",
)
@click.option(
    "--learning_rate",
    default=0.0002,
    help="Learning rate",
)
@click.option(
    "--k_disc_steps",
    default=1,
    help="Number of training step to update only discriminator",
)
@click.option(
    "--display_step",
    default=500,
    help="Number of iterations between each training stats display",
)
def train_mnist_gan(data_dir, models_dir, batch_size, z_dims, loss, n_epochs, learning_rate, k_disc_steps, display_step):
    # TODO: Add config file to deal with hyperparameters
    print(locals())
    pipeline = MNISTPipeline(data_dir, models_dir, batch_size, z_dims)
    pipeline.train(
        loss=loss,
        learning_rate=learning_rate,
        k_disc_steps=k_disc_steps,
        n_epochs=n_epochs,
        display_step=display_step
    )
