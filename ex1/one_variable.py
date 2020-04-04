import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 计算代价函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 梯度下降, return theta, cost
def gradientDescent(X, y, theta, alpha, epoch):
    cost = np.zeros(epoch)
    m = X.shape[0]
    for i in range(epoch):
        theta = theta - (alpha / m) * (X * theta.T - y).T * X
        cost[i] = computeCost(X, y, theta)
    return theta, cost


# 正规方程求解theta
def normalEqn(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y  # np.linalg.inv求逆


path = 'D:\\Documents\\machine-Leanring\\machine-learning-ex1\\ex1\\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 插入一列
data.insert(0, 'Ones', 1)

print(data.head())  # 可查看导入的数据项
print(data.describe())  # 查看导入数据的信息

'''
# 可视化查看数据
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8, 5))
plt.show()
'''

cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]  # 取输入向量
y = data.iloc[:, cols - 1:]  # 取输出向量

X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat([0, 0])
alpha = 0.01
epoch = 2000

# 梯度下降求解theta
final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)
print(final_theta)

# 正规方程求解theta
print(normalEqn(X, y).flatten())

x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

'''
# 查看cost曲线
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(epoch), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
'''