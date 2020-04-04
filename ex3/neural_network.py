from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.io import loadmat


# 加载权重
def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']


# 加载数据
def load_data(path):
    data = loadmat(path)
    return data['X'], data['y']


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


theta1, theta2 = load_weight('D:\\Documents\\machine-Leanring\\machine-learning-ex3\\ex3\\ex3weights.mat')

X, y = load_data('D:\\Documents\\machine-Leanring\\machine-learning-ex3\\ex3\\ex3data1.mat')
y = y.flatten()
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

a1 = X

z2 = a1 @ theta1.T  # 第一层偏差计算
z2 = np.insert(z2, 0, 1, axis=1)  # 插入偏差单位
a2 = sigmoid(z2)  # 第一层激活

z3 = a2 @ theta2.T  # 第二层偏差计算
a3 = sigmoid(z3)  # 第二层激活, 由于模型只有两层, 因此最后得到的即为所需结果

y_pred = np.argmax(a3, axis=1) + 1

# 准确度
accuracy = np.mean(y_pred == y)
print('accuracy = {0}%'.format(accuracy * 100))

