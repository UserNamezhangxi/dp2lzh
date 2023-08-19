import torch
import numpy as np
import matplotlib.pylab as plt
import dp2lzh_pytorch as d2l

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
# xyplot(x, y, 'relu')

# y.sum().backward()
# xyplot(x, x.grad, 'grad of relu')

# y = x.sigmoid()
# y.sum().backward()
# xyplot(x, x.grad, 'sigmoid')


# tanh（双曲正切）函数可以将元素的值变换到-1和1之间：
y = x.tanh()
xyplot(x, y, 'tanh')


y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
