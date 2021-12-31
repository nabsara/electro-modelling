import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    Function for visualizing images: Given a tensor of images, number of images
    and size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def plot_gan_losses(d_loss, g_loss):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(d_loss.detach(), label="discriminator loss")
    ax.plot(g_loss.detach(), label="generator loss")
    ax.set_xlabel("n_epochs")
    ax.legend()
    ax.grid(True)
    plt.show()
