import numpy as np

# 定义函数 f(x)
def f(x):
    return np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) + np.exp(-x[0]-0.1)

# 定义函数 f(x) 的梯度 grad_f(x)
def grad_f(x):
    return np.array([
        np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) - np.exp(-x[0]-0.1),
        3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]-3*x[1]-0.1)
    ])

# 定义梯度下降函数 gradient_descent(f, grad_f, x0, learning_rate=0.01, epochs=10000, eps=1e-10)
# 参数说明：
# f：要最小化的目标函数
# grad_f：目标函数的梯度
# x0：优化变量的初始值
# learning_rate：学习率，控制每次更新的步长
# epochs：最大迭代次数
# eps：当两个相邻的迭代点之间的距离小于 eps 时停止迭代
def gradient_descent(f, grad_f, x0, learning_rate=0.01, epochs=100000, eps=1e-10):
    x = x0
    for k in range(epochs):
        # 计算当前点的梯度
        grad = grad_f(x)
        # 计算下一步的迭代点
        x_new = x - learning_rate * grad
        # 计算两个相邻迭代点之间的距离差
        diff = np.linalg.norm(x_new - x)
        # 如果距离差小于 eps，停止迭代
        if diff < eps:
            # print(diff)
            # print(eps)
            break
        else:
            # 否则继续迭代
            x = x_new
    # 返回最终迭代到的位置和对应的函数值（即极小值）
    return x, f(x)

# 定义初始点 x0，并调用梯度下降函数求解最小值
x0 = np.array([0, 0])
x_star, f_star = gradient_descent(f, grad_f, x0)
print(f"x1*={x_star[0]:.10f},x2*= {x_star[1]:.10f}\nf*= {f_star:.10f}")
