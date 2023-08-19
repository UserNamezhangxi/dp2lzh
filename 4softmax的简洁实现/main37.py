import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import dp2lzh_pytorch as d2l

batch_size = 256

# 获取和读取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义和初始化模型
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_input, num_output):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_input, num_output)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


# 因为前面我们数据返回的每个batch样本x的形状为(batch_size, 1, 28, 28),
# 所以我们要先用view()将x的形状转换成(batch_size, 784)才送入全连接层。
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


from collections import OrderedDict

net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

# 使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数。
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, 0)


# softmax和交叉熵损失函数
loss = nn.CrossEntropyLoss()


#定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


# 计算准确率
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
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, loss, 5, batch_size,  None, None, optimizer)


# 预测
X, y = iter(test_iter).__next__()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])