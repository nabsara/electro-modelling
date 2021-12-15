import click
from electro_modelling.cli import train_mnist_gan


@click.group()
def main():
    pass


main.command()(train_mnist_gan)


if __name__ == "__main__":
    main()
