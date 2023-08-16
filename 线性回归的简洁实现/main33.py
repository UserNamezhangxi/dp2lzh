import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn

batch_size = 10

# 生成数集
input_number = 2
number_example = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(number_example, input_number, dtype=torch.float32)  # torch.randn 均匀/标准分布
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 套公式 y = w1*x1 + w2*x2 + b
labels += torch.tensor(np.random.normal(0,0.01, size=labels.size()), dtype=torch.float32) # 生成和labels相同大小的矩阵 相加，使得样本数据更加随机

# print(features[0], labels[0])
# j = torch.tensor(10)
# print("j=", j)
# print(features.index_select(0, index = j), labels.index_select(0, index = j))
#
#
# # 读取数据
# def data_iter(batch_size, features, labels):
#     num_example = len(features)
#     indices = list(range(num_example))
#     random.shuffle(indices) # 样本的读取是随机的
#     for i in range(0, num_example, batch_size):
#         j = torch.LongTensor(indices[i: min(i + batch_size, num_example)])
#         yield features.index_select(0, j), labels.index_select(0, j)

dataset = Data.TensorDataset(features, labels)
date_iter = Data.DataLoader(dataset, batch_size, shuffle=True)



# # 定义模型
# def linreg(X, w, b):
#     return torch.mm(X, w) + b

# 写法一
net = nn.Sequential(
    nn.Linear(input_number, 1)
    # 此处还可以传入其他层
    )

# 写法2
net2 = nn.Sequential()
net2.add_module("linear", nn.Linear(input_number, 1))
# net.add_module ......

#写法3
from collections import OrderedDict
net3 = nn.Sequential(OrderedDict([
    ("linear", nn.Linear(input_number, 1)),
    # ......
]))

#
# # 初始化模型参数
# w = torch.tensor(np.random.normal(0,0.01, (input_number, 1)), dtype = torch.float32)
# b = torch.zeros(1, dtype=torch.float32)
#
# w.requires_grad_(requires_grad=True)
# b.requires_grad_(requires_grad=True)

# 初始化模型参数
torch.nn.init.normal_(net[0].weight, mean=0, std=1) # 正太分布的均值，标准差
torch.nn.init.constant_(net[0].bias, val=0) # 也可以直接修改bias的data: net[0].bias.data.fill_(0)


#
# # 定义损失函数
# def squared_loss(y_hat, y):
#     return (y_hat - y.view(y_hat.size())) ** 2 / 2
#
# PyTorch在nn模块中提供了各种损失函数，这些损失函数可看作是一种特殊的层，
# PyTorch也将这些损失函数实现为nn.Module的子类。
# 我们现在使用它提供的均方误差损失作为模型的损失函数。
loss = nn.MSELoss()


# # 定义优化算法
# def sgd(params, lr, batch_size):
#     for param in params:
#         param.data -= lr * param.grad / batch_size

import torch.optim as optim  # torch.optim模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等
optimizer = optim.SGD(net.parameters(), lr=0.03)


#
# # 训练模型
# lr = 0.03 # 学习效率
# number_epochs = 3 # 训练次数
# net = linreg
# loss = squared_loss

#
# for epoch in range(number_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y).sum() # l是有关小批量X和y的损失
#         l.backward() # 小批量的损失对模型参数求梯度
#         sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
#         # 不要忘了梯度清零
#         w.grad.data.zero_()
#         b.grad.data.zero_()
#
#     train_loss = loss(net(features, w, b), labels)
#     print('epoch %d, loss %f' % (epoch + 1, train_loss.mean().item()))
#
# print(true_w, '\n', w)
# print(true_b, '\n', b)

num_epochs = 3

for epoch in range(num_epochs):
    for X, y in date_iter:
        out_put = net(X)
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l = loss(out_put, y.view(-1, 1)) # 损失函数
        l.backward() # 求导
        # 是PyTorch中优化器对象的一个方法，用于更新模型的参数。在训练深度神经网络的过程中，
        # 我们需要通过反向传播算法计算每一个参数对损失函数的梯度，然后使用优化器更新参数，使得损失函数最小化。
        # 而optimizer.step()方法就是用于执行参数更新的。
        # 更新后的参数可以通过模型对象的.parameters()方法来获取。
        # 需要注意的是，每次调用step()方法之前，我们需要手动将每个参数的梯度清零，以避免梯度累加
        optimizer.step() # 更新参数
    print('epoch %d, loss: %f' % (epoch, l.item()))

print(optimizer.param_groups)

