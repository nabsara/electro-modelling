# -*- coding: utf-8 -*-

"""
Module that defines train and test DataLoaders on MNIST dataset
"""

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def mnist_data_loader(batch_size, data_dir):
    """
    Create the train and test torch DataLoader on MNIST dataset
    Download the data into data_dir local directory if no MNIST
    directory already exists
    Transform the data to torch Tensor and normalize it

    Parameters
    ----------
    batch_size : int
        size of the batch used to create the DataLoader
    data_dir : str
        path to data directory in which to store MNIST data

    Returns
    -------
        MNIST train DataLoader, MNIST test DataLoader
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=True
    )
    test_set = datasets.MNIST(root=data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
