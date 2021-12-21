import click
from electro_modelling.cli import train_mnist_gan,prepare_dataset


@click.group()
def main():
    pass


main.command()(train_mnist_gan)
main.command()(prepare_dataset)


if __name__ == "__main__":
    main()
