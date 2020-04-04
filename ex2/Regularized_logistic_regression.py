import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report


# 创建更多的特征
def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
    return pd.DataFrame(data)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 计算代价
def computeCost(theta, X, y):
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y) * np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)


# 加入正则的代价函数
def costReg(theta, X, y, l=1):
    _theta = theta[1:]  # 不惩罚第一项
    reg = (l / (2 * len(X))) * (_theta @ _theta)
    return computeCost(theta, X, y) + reg


# 计算梯度
def gradient(theta, X, y):
    return (X.T @ (sigmoid(X @ theta) - y)) / len(X)


# 加入正则的梯度
def gradientReg(theta, X, y, l=1):
    reg = (l / len(X)) * theta
    reg[0] = 0  # 不惩罚第一项
    return gradient(theta, X, y) + reg


# 预测结果
def predict(theta, X):
    probability = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]


path = 'D:\\Documents\\machine-Leanring\\machine-learning-ex2\\ex2\\ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

'''
# 可视化数据
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
'''

x1 = data['Test 1'].values
x2 = data['Test 2'].values

# 由于不能够线性拟合, 因此需要加入高次特征
_data = feature_mapping(x1, x2, power=6)
print(_data.head())

X = _data.values
y = data['Accepted'].values
theta = np.zeros(X.shape[1])
result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, 1), messages=0)
print(result)

final_theta = result[0]

# 测试准确度
predictions = predict(final_theta, X)
correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(X)
print(accuracy)

# 采用skearn中的方法来检验
print(classification_report(predictions, y))

# 计算决策边界
x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)  # 从坐标向量中返回坐标矩阵
z = feature_mapping(xx.ravel(), yy.ravel(), 6).values
z = z @ final_theta
z = z.reshape(xx.shape)

# 绘制数据散点图以及决策边界
positive = data[data.Accepted.isin(['1'])]
negetive = data[data.Accepted.isin(['0'])]
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, marker='o', c='b', label='Accepted')
ax.scatter(negetive['Test 1'], negetive['Test 2'], s=50, marker='x', c='r', label='Rejected')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=2)
ax.set_xlabel('Test 1 score')
ax.set_ylabel('Test 2 score')
plt.contour(xx, yy, z, 0)
plt.ylim(-.8, 1.2)
plt.show()
