import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report


# 逻辑函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def computeCost(theta, X, y):
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y) * np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)


# 计算梯度
def gradient(theta, X, y):
    return (X.T @ (sigmoid(X @ theta) - y)) / len(X)


# 预测结果
def predict(theta, X):
    probability = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]


path = 'D:\\Documents\\machine-Leanring\\machine-learning-ex2\\ex2\\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
data.insert(0, 'Ones', 1)

positive = data[data.admitted.isin(['1'])]
negetive = data[data.admitted.isin(['0'])]

# show the exam1 and exam2
'''
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=2)  # loc:位置  bbox_to_anchor:相对位置  ncol:列数
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.show()
'''

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
theta = np.zeros(X.shape[1])
result = opt.fmin_tnc(func=computeCost, x0=theta, fprime=gradient, args=(X, y), messages=0)

print(result)

# 测试准确度
final_theta = result[0]
predictions = predict(final_theta, X)
correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(X)
print(accuracy)

# 采用skearn中的方法来检验
print(classification_report(predictions, y))

# 绘制决策边界
x1 = np.arange(130, step=0.1)
x2 = -(final_theta[0] + x1 * final_theta[1]) / final_theta[2]
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x1, x2)
ax.set_xlim(0, 130)
ax.set_ylim(0, 130)
ax.set_xlabel('x1')
ax.set_ylabel('y1')
ax.set_title('Decision Boundary')
plt.show()
