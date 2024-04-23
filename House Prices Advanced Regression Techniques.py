import numpy as np
import pandas as pd
import torch
from torch import nn
from Utility.Synthetic_data import load_array
import Utility.Visualize as UV
from Utility.Visualize import plot


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, 1))
    return net


def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    loss = nn.MSELoss()
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs,
          learning_rate, weight_decay, batch_size):
    loss = nn.MSELoss()
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            L = loss(net(X), y)
            L.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse',
                 xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f'{i + 1}, train log rmse{float(train_ls[-1]):f}, '
              f'valid log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_featrues, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log_rmse', xlim=[1, num_epochs], yscale='log')
    print(f'Train log rmse: {float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    train_file_path = '.\\DataSet\\House prices advanced regression techniques\\train.csv'
    test_file_path = '.\\DataSet\\House prices advanced regression techniques\\test.csv'
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    print(train_data.shape)
    print(test_data.shape)
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    numberic_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numberic_features] = all_features[numberic_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numberic_features] = all_features[numberic_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)

    n_train = train_data.shape[0]
    print(n_train)
    train_features = torch.tensor(all_features[:n_train].values.astype(float), dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values.astype(float), dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    in_features = train_features.shape[1]
    print(in_features)

    k, num_epochs, lr, weight_decay, batch_size = 5, 200, 0.01, 0, 64
    #train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    #print(f'{k} folds, mean train log rmse{float(train_l):f}, '
    #      f'mean valid log rmse{float(valid_l):f}')
    #UV.plt.show()

    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
    UV.plt.show()
