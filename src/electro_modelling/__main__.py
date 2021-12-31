import click
from electro_modelling.cli import train_mnist_gan, prepare_dataset, train_techno_gan


@click.group()
def main():
    pass


main.command()(train_mnist_gan)
main.command()(prepare_dataset)
main.command()(train_techno_gan)


if __name__ == "__main__":
    main()
