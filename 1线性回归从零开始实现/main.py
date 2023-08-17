import torch
import numpy as np
import random

# 生成数集
input_number = 2
number_example = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(number_example, input_number, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0,0.01, size=labels.size()), dtype=torch.float32)

print(features[0], labels[0])
j = torch.tensor(10)
print("j=", j)
print(features.index_select(0, index = j), labels.index_select(0, index = j))

# 读取数据
def data_iter(batch_size, features, labels):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices) # 样本的读取是随机的
    for i in range(0, num_example, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_example)])
        yield features.index_select(0, j), labels.index_select(0, j)

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break


# 初始化模型参数
w = torch.tensor(np.random.normal(0,0.01, (input_number, 1)), dtype = torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 训练模型
lr = 0.03 # 学习效率
number_epochs = 3 # 训练次数
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(number_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum() # l是有关小批量X和y的损失
        l.backward() # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()

    train_loss = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_loss.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)


#输出：
#
#tensor([0.2739, 0.9908]) tensor(1.3635)
#j= tensor(10)
#tensor([[-1.0604, -0.6320]]) tensor([4.2445])
#epoch 1, loss 0.037475
#epoch 2, loss 0.000147
#epoch 3, loss 0.000051
#[2, -3.4] 
# tensor([[ 1.9988],
#        [-3.4004]], requires_grad=True)
#4.2 
# tensor([4.1990], requires_grad=True)
#


#
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# print(x.grad_fn)
#
# y = x + 2
# print(y)
# print(y.grad_fn)
#
# print(x.is_leaf, y.is_leaf)
#
# z = y * y * 3
# out = z.mean()
# print(z, out)
#
#
# out.backward()
# print("dz/dx:", x.grad)
#
# out2 = x.sum()
# x.grad.data.zero_()
# out2.backward()
# print("out2:", out2)
# print("dz/dx:", x.grad)
#
# out3 = x.sum()
# x.grad.data.zero_()
# out3.backward()
# print("dz/dx:", x.grad)


######################中断梯度追踪的例子###########################
# x = torch.tensor(1.0,requires_grad=True)
# y1 = x**2
# with torch.no_grad():
#     y2 = x**3
# y3 = y1 + y2
#
# print(x.requires_grad)
# print(y1 ,y1.requires_grad)
# print(y2 ,y2.requires_grad)
# print(y3 ,y3.requires_grad)
#
#
# y3.backward()
# print(x.grad)


# x = torch.tensor(1.0,requires_grad=True)
# print("x:", x)
# print("x.data:",x.data)
# print(x.data.requires_grad)
#
# y = 2 * x
#
# x.data *= 100
# print("x:", x)
# y.backward()
# print(x.grad)









