import sys
import torch
import pandas
import numpy
import numpy as np
import torchvision
import shutil, os
from torchvision import transforms
from torch.utils import data
from SparseIntegratedBlock import ApaFeatureNet
from AdaptivePolynomialApproximator import evaluate
from CIFAR_TEN import load_data_cifar_10
from PIL import Image


data_dir = './DataSet/CIFAR-10-Kaggle'

if __name__ == '__main__':
    net = torch.load('./DataSet/CIFAR-10-Kaggle/ApaFeatureNet.pth', map_location=torch.device(f'cuda:{0}'))
    '''
    labels_info = np.array(pandas.read_csv('./DataSet/CIFAR-10-Kaggle/trainLabels.csv'))
    img_path_prefix = './DataSet/CIFAR-10-Kaggle/train'
    img_save_prefix = './DataSet/CIFAR-10-Kaggle/train_rearranged'
    if not os.path.exists(img_save_prefix):
        os.mkdir(img_save_prefix)
    for item in labels_info:
        img_source_path = img_path_prefix + '/' + str(item[0]) + '.png'
        img_save_path = img_save_prefix + '/' + item[1] + '/' + str(item[0]) + '.png'
        if not os.path.exists(img_save_prefix + '/' + item[1]):
            os.mkdir(img_save_prefix + '/' + item[1])
        shutil.copy(img_source_path, img_save_path)
    test_prefix = './DataSet/CIFAR-10-Kaggle/test/unknown/'
    for img_idx in range(100000):
        old_name = test_prefix + str(img_idx + 1) + '.png'
        new_name = test_prefix + str(img_idx + 1).zfill(6) + '.png'
        os.rename(old_name, new_name)
    '''
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    '''
    dataset = torchvision.datasets.ImageFolder(
        root='./DataSet/CIFAR-10-Kaggle/train_rearranged',
        transform=trans
    )
    data_iter = data.DataLoader(dataset, batch_size=32, shuffle=False)
    print(evaluate(net, data_iter, torch.device(f'cuda:{0}')))
    '''
    dataset = torchvision.datasets.ImageFolder(
        root='./DataSet/CIFAR-10-Kaggle/test',
        transform=trans
    )
    data_iter = data.DataLoader(dataset, batch_size=64, shuffle=False)
    net.eval()
    cnt = 1
    lst = []
    column = ['id', 'label']
    label_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    for X_original, _ in data_iter:
        X = X_original.to(torch.device(f'cuda:{0}'))
        with torch.no_grad():
            y_hat = net(X).argmax(dim=1).to('cpu').numpy()
            for item in y_hat:
                lst.append([cnt, label_names[item]])
                cnt += 1
    answer = pandas.DataFrame(columns=column, data=lst)
    answer.to_csv('./DataSet/CIFAR-10-Kaggle/submission.csv', index=False)
