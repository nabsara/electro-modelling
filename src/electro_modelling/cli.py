import click
from electro_modelling.config import settings
from electro_modelling.pipelines.mnist_pipeline import MNISTPipeline


@click.argument("model")
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
@click.option('--show', is_flag=True)
def train_mnist_gan(model, data_dir, models_dir, batch_size, z_dims, n_epochs, learning_rate, k_disc_steps, display_step, show):
    """
    CLI to train a specified model on MNIST dataset given the input hyperparameters.

    model: str
        model to run : 'dcgan' (SimpleDCGAN), 'hgan' (HingeGAN), 'lsgan' (LeastSquareGAN), 'wgan' (WGAN-GP)
    """
    # TODO: Add config file to deal with hyperparameters
    # TODO: connect to tensorboard
    print(locals())
    pipeline = MNISTPipeline(model, data_dir, models_dir, batch_size, z_dims)
    pipeline.train(
        learning_rate=learning_rate,
        k_disc_steps=k_disc_steps,
        n_epochs=n_epochs,
        display_step=display_step,
        show_fig=show
    )
