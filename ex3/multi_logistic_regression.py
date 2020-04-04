from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


# 加载数据
def loadData(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X, y


# 随机打印一张图片
def plot_an_image(X):
    pick_one = np.random.randint(0, 5000)  # 随机选择一行
    image = X[pick_one, :]  # 从X中抽取图像数据
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap='gray_r')  # 灰度图
    plt.xticks([])  # 去除刻度，美观
    plt.yticks([])
    print('this should be {}'.format(y[pick_one]))
    plt.show()


# 随机打印100张图
def plot_100_image(X):
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_img = X[sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    for row in range(10):
        for col in range(10):
            ax_array[row, col].matshow(sample_img[10 * row + col].reshape((20, 20)), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 正则化的代价函数
def regularized_cost(theta, X, y, l):
    reg = theta[1:]
    first = -y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta))
    reg = (reg @ reg) * l / (2 * len(X))
    return np.mean(first) + reg


# 正则化的梯度下降
def regularized_gradient(theta, X, y, l):
    reg = theta[1:]
    first = (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    reg = np.concatenate([np.array([0]), (1 / len(X)) * reg])  # concatenate 数组拼接函数, 此处在前面加一个0, 即第一项不惩罚
    return first + reg


# 一对多分类训练
def one_vs_all(X, y, l, K):
    all_theta = np.zeros((K, X.shape[1]))
    for i in range(1, K + 1):
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label == i else 0 for label in y])  # 对于第i类, 标签为i则y_i为1, 否则y_i为0
        ret = minimize(fun=regularized_cost, x0=theta, args=(X, y_i, l), method='TNC', jac=regularized_gradient,
                       options={'disp': True})  # disp为True, 则打印详细的迭代信息;
        all_theta[i - 1, :] = ret.x
    return all_theta


def predict_all(X, all_theta):
    h = sigmoid(X @ all_theta.T)
    h_argmax = np.argmax(h, axis=1)  # 返回array中数值最大数的下标
    h_argmax += 1
    return h_argmax


X, y = loadData('D:\\Documents\\machine-Leanring\\machine-learning-ex3\\ex3\\ex3data1.mat')
print(np.unique(y))  # 查看有几类标签
print(X.shape, y.shape)

# plot_an_image(X)
# plot_100_image(X)

X = np.insert(X, 0, 1, axis=1)
y = y.flatten()  # 这里消除一个维度，方便后面的计算

# 训练
all_theta = one_vs_all(X, y, 1, 10)

# 预测
y_pred = predict_all(X, all_theta)

# 计算准确度
accuracy = np.mean(y_pred == y)
print('accuracy = {0}%'.format(accuracy * 100))
