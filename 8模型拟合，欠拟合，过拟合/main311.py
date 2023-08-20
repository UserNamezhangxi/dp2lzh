import torch
import numpy as np
import dp2lzh_pytorch as d2l

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
feature = torch.randn(n_train + n_test, 1)
poly_features = torch.cat((feature, torch.pow(feature, 2), torch.pow(feature, 3)), 1)
labels = true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b

labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()))

print(feature[:2])
print(labels[:2])


def semilogy(x_value, y_value, x_label, y_label, x2_value=None, y2_value=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_value, y_value)
    if x2_value and y2_value:
        d2l.plt.semilogy(x2_value, y2_value, linestyle=':')
        d2l.plt.legend(legend)

    d2l.plt.show()


num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)

        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print("final epoch train loss:", train_ls[-1] , "test loss : " , test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])

    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)


# 三阶多项式 正常拟合
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])


# 线性函数拟合（欠拟合）
fit_and_plot(feature[:n_train, :], feature[n_train:, :], labels[:n_train], labels[n_train:])


# 三阶多项式 训练样本不足，过拟合
fit_and_plot(poly_features[:3, :], poly_features[n_train:, :], labels[:3], labels[n_train:])