import torch
import numpy as np
import matplotlib.pylab as plt
import dp2lzh_pytorch as d2l

number_input, number_output, number_hidden = 784, 10, 256

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 初始化模型参数
W1 = torch.tensor(np.random.normal(0,0.01, size=(number_input, number_hidden)), dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0,0.01, size=(number_hidden, number_output)), dtype=torch.float, requires_grad=True)

b1 = torch.zeros(number_hidden, dtype=torch.float, requires_grad=True)
b2 = torch.zeros(number_output, dtype=torch.float, requires_grad=True)

# 定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

# 定义模型
def net(X):
    X = X.view((-1, number_input))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 训练模型
num_epochs = 5

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
    params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()

            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f" % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc))

num_epochs, lr = 5, 100.0
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, [W1, b1, W2, b2], lr, None)