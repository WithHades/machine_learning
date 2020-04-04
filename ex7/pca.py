from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


# 特征标准化
def featureNormalize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    X_norm = (X - means) / stds
    return X_norm, means, stds


def pca(X):
    sigma = (X.T @ X) / len(X)
    U, S, V = np.linalg.svd(sigma)  # SVD奇异值分解
    return U, S, V


# 将数据投影到主成分U上
def projectData(X, U, K):
    Z = X @ U[:, :K]
    return Z


# 将数据恢复到高维空间
def recoverData(Z, U, K):
    X_rec = Z @ U[:, :K].T
    return X_rec


def displayData(X, row, col):
    fig, axs = plt.subplots(row, col, figsize=(8, 8))
    for r in range(row):
        for c in range(col):
            axs[r][c].imshow(X[r * col + c].reshape(32, 32).T, cmap='Greys_r')  # cmap代表色彩盘,0通道的灰度图
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])


mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex7\\ex7\\ex7data1.mat')
X = mat['X']

# plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
# plt.show()

X_norm, means, stds = featureNormalize(X)
U, S, V = pca(X_norm)
print(U)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')

plt.plot([means[0], means[0] + 1.5 * S[0] * U[0, 0]],
         [means[1], means[1] + 1.5 * S[0] * U[0, 1]],
         c='r', linewidth=3, label='first Principal Component')

plt.plot([means[0], means[0] + 1.5 * S[1] * U[1, 0]],
         [means[1], means[1] + 1.5 * S[1] * U[1, 1]],
         c='g', linewidth=3, label='Second Principal Component')

plt.grid()
plt.axis('equal')  # x轴与y轴一样长
plt.legend()
plt.show()

# 将数据投影到主成分U上
Z = projectData(X_norm, U, 1)

# 将数据恢复到高维空间,但是恢复只能恢复近似值
X_rec = recoverData(Z, U, 1)

plt.figure(figsize=(7, 5))
plt.axis("equal")

# 正常数据
plt.scatter(X_norm[:, 0], X_norm[:, 1], s=30, facecolors='none',
            edgecolors='b', label='Original Data Points')

# 恢复之后的数据
plt.scatter(X_rec[:, 0], X_rec[:, 1], s=30, facecolors='none',
            edgecolors='r', label='PCA Reduced Data Points')

plt.title("Example Dataset: Reduced Dimension Points Shown", fontsize=14)
plt.xlabel('x1 [Feature Normalized]', fontsize=14)
plt.ylabel('x2 [Feature Normalized]', fontsize=14)
plt.grid(True)

# 将原始点和恢复后的点对应起来
for x in range(X_norm.shape[0]):
    plt.plot([X_norm[x, 0], X_rec[x, 0]], [X_norm[x, 1], X_rec[x, 1]], 'k--')
plt.legend()
plt.show()

# 将PCA应用到人脸图像压缩上
mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex7\\ex7\\ex7faces.mat')
X = mat['X']

displayData(X, 10, 10)
plt.show()

X_norm, means, stds = featureNormalize(X)
U, S, V = pca(X_norm)

# 看一下主成分U
displayData(U[:, :36].T, 6, 6)
plt.show()

# 将数据投影到主成分U上
Z = projectData(X_norm, U, 36)

# 将数据恢复到高维空间,但是恢复只能恢复近似值
X_rec = recoverData(Z, U, 36)
displayData(X_rec, 10, 10)
plt.show()
