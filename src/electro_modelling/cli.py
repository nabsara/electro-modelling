import click
import os
from electro_modelling.config import settings
from electro_modelling.pipelines.mnist_pipeline import MNISTPipeline
from electro_modelling.pipelines.dataset_pipeline import TechnoDatasetPipeline
from electro_modelling.pipelines.techno_pipeline import TechnoPipeline


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




@click.option(
    "--nfft",
    default=1024,
    help="Nfft",
)
@click.option(
    "--sr",
    default=16000,
    help="Sampling Rate",
)
@click.option(
    "--data_path",
    default="/fast-1/tmp/",
    help="Sampling Rate",
)
@click.option(
    "--save_dir",
    default="/slow-2/atiam/electro_modelling",
    help="Sampling Rate",
)
@click.option(
    "--nb_samples",
    default=None,
    help="Sampling Rate",
)
def prepare_dataset(nfft, sr, data_path, save_dir, nb_samples):
    # Signal processing parameters
    nmels = int(nfft/2)
    # File locations
    dataset_location = os.path.join(data_path, 'techno.dat')
    save_location = os.path.join(save_dir, 'techno_spectrograms.pkl')
    # Instanciate pipeline
    pipeline = TechnoDatasetPipeline(nfft, nmels, sr, dataset_location=dataset_location, save_location=save_location)
    pipeline.process_dataset(nb_samples=nb_samples)


@click.argument("model")
@click.option(
    "--dataset_file",
    default="techno_spectrograms_nb_samples_64.pkl",
    help="Name of the dataset pickle file",
)
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
    default=32,
    help="Data loader batch size",
)
@click.option(
    "--z_dims",
    default=256,
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
def train_techno_gan(model, dataset_file, data_dir, models_dir, batch_size, z_dims, n_epochs, learning_rate, k_disc_steps, display_step, show):
    """
    CLI to train a specified model on MNIST dataset given the input hyperparameters.

    model: str
        model to run : 'dcgan' (SimpleDCGAN), 'hgan' (HingeGAN), 'lsgan' (LeastSquareGAN), 'wgan' (WGAN-GP)
    """
    # TODO: Add config file to deal with hyperparameters
    # TODO: connect to tensorboard
    print(locals())
    pipeline = TechnoPipeline(
        model_name=model,
        data_dir=os.path.join(data_dir, dataset_file),
        models_dir=models_dir,
        batch_size=batch_size,
        z_dims=z_dims,
        phase_method='griff'
    )
    pipeline.train(
        learning_rate=learning_rate,
        k_disc_steps=k_disc_steps,
        n_epochs=n_epochs,
        display_step=display_step,
        show_fig=show
    )
