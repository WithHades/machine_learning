import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import imageio


# 对每个样本点, 寻找与其距离最近的分类中心点
def findClosestCentroids(X, centroids):
    idx = []
    max_dist = 1000000  # 设置的最大距离
    for i in range(len(X)):
        minus = X[i] - centroids
        dist = np.diag(minus @ minus.T)
        if dist.min() < max_dist:
            ci = np.argmin(dist)
            idx.append(ci)
    return np.array(idx)


# 计算每个簇的质心
def computeCentroids(X, idx):
    centroids = []
    for i in range(len(np.unique(idx))):
        u_k = X[idx == i].mean(axis=0)  # 对每个簇的点求均值
        centroids.append(u_k)
    return np.array(centroids)


# 绘制数据点、中心点以及中心点的迭代路径
def plot_data(X, centroids, idx=None, title=''):
    colors = ['b', 'g', 'gold', 'darkorange', 'salmon', 'olivedrab',
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro',
              'coral', 'aliceblue', 'dimgray', 'mintcream',
              'mintcream']

    # assert 断言,如果表达式为false,则触发异常.
    assert len(centroids[0]) <= len(colors), 'colors not enough'

    # 如果idx存在，就根据idx把X分类
    subX = []
    if idx is not None:
        for i in range(centroids[0].shape[0]):
            x_i = X[idx == i]
            subX.append(x_i)
    else:
        subX = [X]

    # 绘制分类后的X点，不同类别X采用不同颜色
    plt.figure(figsize=(8, 5))
    for i in range(len(subX)):
        xx = subX[i]
        plt.scatter(xx[:, 0], xx[:, 1], c=colors[i], label='Cluster %d' % i)

    plt.legend()
    plt.grid(True)
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('y1', fontsize=14)
    plt.title('plot of x points' + title, fontsize=16)

    # 绘制分类中心点
    xx, yy = [], []
    for centroid in centroids:
        xx.append(centroid[:, 0])
        yy.append(centroid[:, 1])
    plt.plot(xx, yy, 'rx--', markersize=8)  # 第三个参数'[color][marker][line]'


# 进行指定次数的迭代，不断的求解最优的中心点
def runKmeans(X, centroids, max_iters):
    centroids_all = [centroids]
    centroid_i = centroids

    # 对于每次迭代，首先对每个样本点，计算距离最近的中心点；然后计算中心点的质心。
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroid_i)
        centroid_i = computeCentroids(X, idx)
        centroids_all.append(centroid_i)
    return idx, centroids_all


# 随机化初始中心点，K个初始化中心点从X中随机选择
def initCentroids(X, K):
    idx = np.random.choice(X.shape[0], K)  # 从X中选择三个索引值
    centroids = X[idx]
    return centroids


# 加载数据
mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex7\\ex7\\ex7data2.mat')
X = mat['X']

# 指定一个中心点列表
init_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# 计算分别对每个点计算，求离哪个中心点近
idx = findClosestCentroids(X, init_centroids)

# 每个簇求解中心点
# print(computeCentroids(X, idx))

plot_data(X, [init_centroids], idx, '->init_centroids')
plt.show()

# 迭代20次
idx, centroids_all = runKmeans(X, init_centroids, 20)
plot_data(X, centroids_all, idx, '->iters=20')
plt.show()

# 随机生成三次中心点，查看Kmeans效果
for i in range(3):
    centroids = initCentroids(X, 3)
    idx, centroids_all = runKmeans(X, centroids, 10)
    plot_data(X, centroids_all, idx, '->The result of the initCentroids')
    plt.show()


# 采用Kmeans对图像进行压缩
A = imageio.imread('D:\\Documents\\machine-Leanring\\machine-learning-ex7\\ex7\\bird_small.png')
# A为128*128*3,表示128*128像素,通道为3

# plt.imshow(A)
# plt.show()

A = A / 255  # 将A的值变为0-1之间
X = A.reshape(-1, 3)
K = 8  # 将图片颜色压缩到16种
centroids = initCentroids(X, K)  # 随机初始化16种颜色
idx, centroids_all = runKmeans(X, centroids, 10)  # 20次迭代选择最好的16种颜色

# 重新生成一张图
img = np.zeros(X.shape)
centroids = centroids_all[-1]  # 取最后一次数据即可
for i in range(len(centroids)):
    img[idx == i] = centroids[i]  # i点属于centroids[i]类别,则将其颜色设置为centroids[i]

# 维度还原
img = img.reshape((128, 128, 3))
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title('original')
ax[0].imshow(A)
ax[1].set_title('compress')
ax[1].imshow(img)
plt.show()
