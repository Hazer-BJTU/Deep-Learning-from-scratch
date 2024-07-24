import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import Utility.Visualize as Uv
from Utility.Visualize import show_images


def get_cifar10_labels(labels):
    text_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return [text_label[int(i)] for i in labels]


def get_mnist_labels(labels):
    return [int(i) for i in labels]


def load_data_cifar_10(batch_size, resize):
    trans = [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    cifar10_train = torchvision.datasets.CIFAR10(
        root=".\\DataSet", train=True, transform=trans, download=True
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root=".\\DataSet", train=False, transform=trans, download=True
    )
    return (data.DataLoader(cifar10_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(cifar10_test, batch_size, shuffle=True, num_workers=4))


if __name__ == '__main__':
    Uv.use_svg_display()

    trans = transforms.ToTensor()
    cifar10_train = torchvision.datasets.CIFAR10(
        root=".\\DataSet", train=True, transform=trans, download=True
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root=".\\DataSet", train=False, transform=trans, download=True
    )

    X, y = next(iter(data.DataLoader(cifar10_train, batch_size=18)))
    show_images(X.reshape(18, 3, 32, 32), 2, 9, titles=get_cifar10_labels(y))
    Uv.plt.show()
