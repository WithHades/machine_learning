import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm


def plot_data():
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()


# 核函数-高斯函数,但是实际上我们直接调用SVC中的高斯核函数即可
def gaussKernel(x1, x2, sigma):
    return np.exp(-((x1 - x2) ** 2).sum() / (2 * sigma ** 2))


# 绘制边界
def plotBoundary(clf, X):
    x_min, x_max = X[:, 0].min() * 1.2, X[:, 1].max() * 1.1
    y_min, y_max = X[:, 0].min() * 1.1, X[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))  # 将原始数据变成网格数据形式
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)  # contour绘制矩阵的等高线。


mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex6\\ex6\\ex6data2.mat')
X = mat['X']
y = mat['y']

sigma = 0.1
gamma = np.power(sigma, -2) / 2

# C:惩罚参数
# kernel='rbf'时，为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但可能会过拟合
# gamma:'rbf'的核函数参数。默认是’auto’，则会选择1/n_features
clf = svm.SVC(C=1, kernel='rbf', gamma=gamma)
modle = clf.fit(X, y.flatten())
plot_data()
plotBoundary(modle, X)
plt.show()


# 通过一个案例说明,如何寻找最优的C和sigma
# 加载数据
mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex6\\ex6\\ex6data3.mat')
X, y = mat['X'], mat['y']
Xval, yval = mat['Xval'], mat['yval']

Cvalues = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
sigmavalues = Cvalues
best_pair, best_score = (0, 0), 0

for C in Cvalues:
    for sigma in sigmavalues:
        gamma = np.power(sigma, -2.) / 2
        clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        clf.fit(X, y.flatten())
        score = clf.score(Xval, yval)  # Return the mean accuracy on the given test data and labels.
        if score > best_score:
            best_score = score
            best_pair = (C, sigma)

print('best_pair={}, best_score={}'.format(best_pair, best_score))

# 展示一下最优C和sigma的效果
gamma = np.power(best_pair[1], -2) / 2
clf = svm.SVC(C=best_pair[0], kernel='rbf', gamma=gamma)
clf.fit(X, y.flatten())
plot_data()
plotBoundary(clf, X)
plt.show()
