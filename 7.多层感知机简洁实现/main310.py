import torch
import torch.nn as nn
import dp2lzh_pytorch as d2l

number_input, number_output, number_hidden = 784, 10, 256

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 因为前面我们数据返回的每个batch样本x的形状为(batch_size, 1, 28, 28),
# 所以我们要先用view()将x的形状转换成(batch_size, 784)才送入全连接层。
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# 定义模型
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(number_input, number_hidden),
    nn.ReLU(),  # 激活函数直接这样
    nn.Linear(number_hidden, number_output),
)

for param in net.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)


# 定义损失函数
loss = torch.nn.CrossEntropyLoss()


# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)


# 计算测试样本的识别准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# 训练模型
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
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, lr, optimizer)