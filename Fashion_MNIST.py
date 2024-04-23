import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import Utility.Visualize as UV
from Utility.Visualize import show_images

def get_fashion_mnist_labels(labels):
    text_label = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_label[int(i)] for i in labels]


def get_mnist_labels(labels):
    return [int(i) for i in labels]


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=".\\DataSet", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=".\\DataSet", train=False, transform=trans, download=True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=4))


if __name__ == '__main__':
    UV.use_svg_display()

    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root=".\\DataSet", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=".\\DataSet", train=False, transform=trans, download=True
    )

    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    UV.plt.show()
