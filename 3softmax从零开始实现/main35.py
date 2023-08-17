import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import dp2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

# torch.Size([1, 28, 28]) 9


# 展示数据
# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

# 小批量获取数据
batch_size = 256
if sys.platform.startswith("win"):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 由于图像有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为
# 784×10 和 1×10 的矩阵

input_number = 784 # 每一个图的大小是28*28 px 模型的输入向量的长度是 28×28=784
out_number = 10 # 输出有10种类别

# 初始化模型参数,并设置梯度
W = torch.tensor(np.random.normal(0, 0.01, size=(input_number, out_number)), dtype=torch.float, requires_grad=True)
b = torch.zeros(out_number, dtype=torch.float, requires_grad=True)


# 定义softmax 运算
def softmax(X):
    exp_X = X.exp()  # e^x(i) 次方
    partition = exp_X.sum(dim=1, keepdims=True)
    return exp_X / partition


# 定义模型
def net(X):
    return softmax(torch.mm(X.view(-1, input_number), W) + b)


# 定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 3.6.6 计算分类准确率
# 其中y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同。
# 相等条件判断式(y_hat.argmax(dim=1) == y)是一个类型为ByteTensor的Tensor，
# 我们用float()将其转换为值为0（相等为假）或1（相等为真）的浮点型Tensor。

def accuracy(y_hat, y):
    print("y_hat.argmax(dim=1):",y_hat.argmax(dim=1))  # tensor([2, 2])
    print("y" , y)  # tensor([0, 2])
    print("(y_hat.argmax(dim=1) == y).float()", (y_hat.argmax(dim=1) == y).float())  # tensor([0., 1.])
    print("(y_hat.argmax(dim=1) == y).float().mean", (y_hat.argmax(dim=1) == y).float().mean())  # tensor(0.5000)
    return (y_hat.argmax(dim=1) == y).float().mean().item()  #0.5

# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([0, 2])
# y_hat.gather(1, y.view(-1, 1))
# print(accuracy(y_hat, y))  # 用y_hat 中最大的元素下标和y进行逐个对比，如果一样就 为1，否则为，最后统计 为1 的概率就是准确率


# 计算分类准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# print(evaluate_accuracy(test_iter, net))


# 训练模型
num_epochs, lr = 5, 0.1

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
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

# 训练前参数是初始化的默认值
print("1W:", W)
print("1b:", b)

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 此时已经训练出来了 w 和 b ，net 中进行预测的时候，就有值了
print("2W:", W)
print("2b:", b)


# 预测
X, y = iter(test_iter).__next__()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
