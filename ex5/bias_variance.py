import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt


def plot_Data():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 1:], y, c='r', marker='x')
    plt.grid(True)  # 设置是否显示网格线


# 正则代价函数
def costReg(theta, X, y, l):
    cost = ((X @ theta - y.flatten()) ** 2).sum()
    regterm = l * (theta[1:] @ theta[1:])
    return (cost + regterm) / (2 * len(X))


# 正则梯度下降
def gradientReg(theta, X, y, l):
    grad = (X @ theta - y.flatten()) @ X
    regterm = l * theta
    regterm[0] = 0
    return (grad + regterm) / len(X)


# 拟合线性回归
def trainLinearReg(X, y, l):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=costReg,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=gradientReg)
    return res.x


# 画出学习曲线，即交叉验证误差和训练误差随样本数量的变化的变化
def plot_learning_curve(X, y, Xval, yval, l):
    x = range(1, len(X) + 1)
    training_cost, cv_cost = [], []
    for i in x:
        res = trainLinearReg(X[:i], y[:i], l)  # 样本数量不断增加
        training_cost_i = costReg(res, X[:i], y[:i], 0)
        cv_cost_i = costReg(res, Xval, yval, 0)
        training_cost.append(training_cost_i)
        cv_cost.append(cv_cost_i)
    plt.figure(figsize=(8, 5))
    plt.plot(x, training_cost, label='training cost', c='r')
    plt.plot(x, cv_cost, label='cv cost', c='b')
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Learning curve for linear regression:lamda=' + str(l))
    plt.grid(True)


# 添加多项式特征，从二次方开始开始插入
def genPolyFeatures(X, power):
    Xpoly = X.copy()
    for i in range(2, power + 1):
        Xpoly = np.insert(Xpoly, Xpoly.shape[1], np.power(Xpoly[:, 1], i), axis=1)
    return Xpoly


# 获取均值与标准差
def get_means_std(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)  # ddof=1 means 样本标准差
    return means, stds


# 数据标准化
def featureNormalize(myX, means, stds):
    X_norm = myX.copy()
    X_norm[:, 1:] = X_norm[:, 1:] - means[1:]
    X_norm[:, 1:] = X_norm[:, 1:] / stds[1:]
    return X_norm


# 绘制拟合曲线
def plot_fit(means, stds, l):
    theta = trainLinearReg(X_norm, y, l)
    x = np.linspace(-75, 55, 50)
    xmat = x.reshape(-1, 1)
    xmat = np.insert(xmat, 0, 1, axis=1)
    Xmat = genPolyFeatures(xmat, power)
    Xmat_norm = featureNormalize(Xmat, means, stds)
    plot_Data()
    plt.plot(x, Xmat_norm @ theta, 'b--')


data = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex5\\ex5\\ex5data1.mat')

X, y = data['X'], data['y']
Xval, yval = data['Xval'], data['yval']
Xtest, ytest = data['Xtest'], data['ytest']

X = np.insert(X, 0, 1, axis=1)
Xval = np.insert(Xval, 0, 1, axis=1)
Xtest = np.insert(Xtest, 0, 1, axis=1)
print('X={},y={}'.format(X.shape, y.shape))
print('Xval={},yval={}'.format(Xval.shape, yval.shape))
print('Xtest={},ytest={}'.format(Xtest.shape, ytest.shape))

# plot_Data()
# plt.show()

theta = np.ones(X.shape[1])

fit_theta = trainLinearReg(X, y, 0)
plot_Data()
plt.plot(X[:, 1], X @ fit_theta)
plt.show()

power = 6
train_means, train_stds = get_means_std(genPolyFeatures(X, power))
X_norm = featureNormalize(genPolyFeatures(X, power), train_means, train_stds)
Xval_norm = featureNormalize(genPolyFeatures(Xval, power), train_means, train_stds)
Xtest_norm = featureNormalize(genPolyFeatures(Xtest, power), train_means, train_stds)

# 观察不同lambdas下的曲线
plot_fit(train_means, train_stds, 0)
plot_learning_curve(X_norm, y, Xval_norm, yval, 0)
plt.show()

plot_fit(train_means, train_stds, 1)
plot_learning_curve(X_norm, y, Xval_norm, yval, 1)
plt.show()

# 不同lambdas取值下的代价
lambdas = [0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.]
errors_train, errors_val = [], []
for l in lambdas:
    theta = trainLinearReg(X_norm, y, l)
    errors_train.append(costReg(theta, X_norm, y, 0))  # 记得把lambda = 0
    errors_val.append(costReg(theta, Xval_norm, yval, 0))

plt.figure(figsize=(8, 5))
plt.plot(lambdas, errors_train, label='Train')
plt.plot(lambdas, errors_val, label='Cross Validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error')
plt.grid(True)
plt.show()
print('lambda={}'.format(lambdas[np.argmin(errors_val)]))
