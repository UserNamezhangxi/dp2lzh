import torch
import numpy as np
import dp2lzh_pytorch as d2l

n_train, n_test, num_inputs = 20, 100, 200

# 人为构造数据
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
feature = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(feature, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

train_feature, test_feature = feature[:n_train, :], feature[n_train:, :]
train_label, test_label = labels[:n_train], labels[n_train:]


# 定义训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_feature, train_label)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


# 初始化参数
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义l2范数惩罚项
def l2_penalty(w):
    return (w**2).sum() / 2


# 训练
def fit_and_plot(lambd):
    w, b = init_params()
    train_l, test_l = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            d2l.sgd(params=[w, b], lr=lr, batch_size=batch_size)

        train_l.append(loss(net(train_feature, w, b), train_label).mean().item())
        test_l.append(loss(net(test_feature, w, b), test_label).mean().item())

    d2l.semilogy(range(1, num_epochs + 1), train_l, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_l, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())


fit_and_plot(4)