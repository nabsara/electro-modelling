from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def mnist_data_loader(batch_size, data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root=data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
