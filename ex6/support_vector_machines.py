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


def plotBoundary(clf, X):
    x_min, x_max = X[:, 0].min() * 1.2, X[:, 1].max() * 1.1
    y_min, y_max = X[:, 0].min() * 1.1, X[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))  # 将原始数据变成网格数据形式
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)  # contour绘制矩阵的等高线。


mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex6\\ex6\\ex6data1.mat')
X = mat['X']
y = mat['y']

# plot_data()
# plt.show()

models = [svm.SVC(C, kernel='linear') for C in [1, 100]]  # 支持向量机可以使用线性函数
clfs = [model.fit(X, y.ravel()) for model in models]  # ravel将多维数组转为一维

title = ['SVM Decision Boundary with C = {} (Example Dataset 1)'.format(C) for C in [1, 100]]
print(title)
for model, title in zip(clfs, title):
    plot_data()
    plt.title(title)
    plotBoundary(model, X)
plt.show()
