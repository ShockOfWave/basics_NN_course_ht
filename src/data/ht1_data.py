import os
import torch
import torchvision
import torchvision.transforms as transforms
from src.utils import get_project_root


def get_dataset(batch_size=256):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=os.path.join(get_project_root(), 'mnist_data'),
        train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root=os.path.join(get_project_root(), 'mnist_data'),
        train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
