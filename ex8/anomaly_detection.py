import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import stats


def plot_data():
    plt.figure(figsize=(8, 5))
    plt.plot(X[:, 0], X[:, 1], 'bx')


# 获取高斯参数
def getGaussianParams(X):
    mu = X.mean(axis=0)  # 均值
    sigma = np.cov(X.T)  # 求协方差
    return mu, sigma


# 高斯模型
def gaussian(X, mu, sigma):
    norm = 1. / (np.power(2 * np.pi, X.shape[1] / 2) * np.sqrt(np.linalg.det(sigma)))  # np.linalg.det计算行列式
    exp = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        exp[i] = np.exp(-0.5 * (X[i] - mu).T @ np.linalg.inv(sigma) @ (X[i] - mu))  # np.linalg.inv求逆
    return norm * exp


# 绘制等高线
def plotContours(mu, sigma):
    x, y = np.mgrid[0:30:.3, 0:30:.3]
    points = np.c_[x.ravel(), y.ravel()]  # 按行连接
    z = gaussian(points, mu, sigma)
    z = z.reshape(x.shape)

    # 可以调用函数直接进行高斯分布
    # multi_normal = stats.multivariate_normal(mu, sigma)
    # z = multi_normal.pdf(np.dstack((x, y)))

    cont_levels = [10 ** h for h in range(-20, 0, 3)]  # 该变量设置等高线的个数,可以为整数或者类数组.此处由于高斯函数设置等高线个数太密集,所以采用数组形式
    plt.contour(x, y, z, cont_levels)
    plt.title('Gaussian Contours', fontsize=16)


# 根据计算F1值选择合适的epsilons值
def selectThreshold(yval, pval):
    # 计算F1
    def computeF1(yval, pval):
        m = len(yval)
        tp = float(len([i for i in range(m) if pval[i] and yval[i]]))
        fp = float(len([i for i in range(m) if pval[i] and not yval[i]]))
        fn = float(len([i for i in range(m) if not pval[i] and yval[i]]))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return F1

    # 产生1000个epsilons值分别计算F1
    epsilons = np.linspace(min(pval), max(pval), 1000)
    bestF1, bestEpsilon = 0, 0
    for e in epsilons:
        pval_ = pval < e
        thisF1 = computeF1(yval, pval_)
        if thisF1 > bestF1:
            bestF1 = thisF1
            bestEpsilon = e
    return bestF1, bestEpsilon


mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex8\\ex8\\ex8data1.mat')
X = mat['X']
Xval, yval = mat['Xval'], mat['yval']

plot_data()
# plt.show()

plotContours(*getGaussianParams(X))  # *代表有多个参数
plt.show()

# 获取合适的epslion值
mu, sigma = getGaussianParams(X)
pval = gaussian(Xval, mu, sigma)
bestF1, bestEpslion = selectThreshold(yval, pval)
print('bestF1={},bestEpslion={}'.format(bestF1, bestEpslion))

# 筛选离群点
y = gaussian(X, mu, sigma)
x = np.array([X[i] for i in range(len(y)) if y[i] < bestEpslion])

# 绘图
plot_data()
plotContours(mu, sigma)
plt.scatter(x[:, 0], x[:, 1], s=80, facecolor='none', edgecolors='r')
plt.title('mark the negatives')
plt.show()
