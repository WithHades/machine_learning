import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 代价函数
def computeCost(X, y, theta):
    inner = np.power((X*theta.T - y), 2)
    return np.sum(inner) / (2*(len(X)))


# 梯度下降
def gradientDescent(X, y, theta, alpha, epoch):
    cost = np.zeros(epoch)
    m = X.shape[0]
    for i in range(epoch):
        theta = theta - (alpha / m) * (X * theta.T - y).T * X
        cost[i] = computeCost(X, y, theta)
    return theta, cost


path = 'D:\\Documents\\machine-Leanring\\machine-learning-ex1\\ex1\\ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data = (data - data.mean()) / data.std()  # 数据归一化
data.insert(0, 'Ones', 1)
print(data.head())

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:]

X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat([0, 0, 0])
alpha = 0.01
epoch = 1000

final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)
print(final_theta)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(epoch), cost, 'r', label='Iterations')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

