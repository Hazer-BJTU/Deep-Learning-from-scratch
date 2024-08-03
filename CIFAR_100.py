import torch
import torchvision
from torch.utils import data
from torchvision import transforms


def get_mnist_labels(labels):
    return [int(i) for i in labels]


def load_data_cifar_100(batch_size, resize):
    trans_train = [ 
        torchvision.transforms.RandomResizedCrop(32, scale=(0.25, 1), ratio=(0.8, 1.25)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), transforms.Normalize([0.51, 0.49, 0.44], [0.27, 0.26, 0.28])
    ]
    trans_test = [transforms.ToTensor(), transforms.Normalize([0.51, 0.49, 0.44], [0.27, 0.26, 0.28])]
    if resize:
        trans_train.insert(0, transforms.Resize(resize))
        trans_test.insert(0, transforms.Resize(resize))
    trans_train = transforms.Compose(trans_train)
    trans_test = transforms.Compose(trans_test)
    cifar100_train = torchvision.datasets.CIFAR100(
        root=".\\DataSet", train=True, transform=trans_train, download=True
    )
    cifar100_test = torchvision.datasets.CIFAR100(
        root=".\\DataSet", train=False, transform=trans_test, download=True
    )
    return (data.DataLoader(cifar100_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(cifar100_test, batch_size, shuffle=True, num_workers=4))


if __name__ == '__main__':
    trans = transforms.ToTensor()
    cifar100_train = torchvision.datasets.CIFAR100(
        root=".\\DataSet", train=True, transform=trans, download=True
    )
    cifar100_test = torchvision.datasets.CIFAR100(
        root=".\\DataSet", train=False, transform=trans, download=True
    )

    X, y = next(iter(data.DataLoader(cifar100_train, batch_size=18)))
