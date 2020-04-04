import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt


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


# 展开X/theta值以便能够传入高级优化方法进行运算
def serialize(X, Theta):
    return np.r_[X.flatten(), Theta.flatten()]


# 还原为原来的X/theta值
def deserialize(seq, nm, nu, nf):
    return seq[:nm * nf].reshape(nm, nf), seq[nm * nf:].reshape(nu, nf)


# 协同过滤下的代价函数
def cofiCostFunc(params, Ynorm, R, nm, nu, nf, l=0):
    X, Theta = deserialize(params, nm, nu, nf)
    error = 0.5 * np.square((X @ Theta.T - Ynorm) * R).sum()  # 之所以*R,是因为我们只要我们评过分数的电影的数据
    reg1 = .5 * l * np.square(Theta).sum()
    reg2 = .5 * l * np.square(X).sum()
    return error + reg1 + reg2


# 协同过滤下的梯度下降函数
def cofiGradient(params, Y, R, nm, nu, nf, l=0):
    X, Theta = deserialize(params, nm, nu, nf)
    X_grad = ((X @ Theta.T - Y) * R) @ Theta + l * X
    Theta_grad = ((X @ Theta.T - Y) * R).T @ X + l * Theta
    return serialize(X_grad, Theta_grad)


# 检查梯度下降是否正常工作
def checkGradient(params, Y, myR, nm, nu, nf, l=0):
    grad = cofiGradient(params, Y, myR, nm, nu, nf, l)
    e = 0.0001
    nparams = len(params)
    e_vec = np.zeros(nparams)

    for i in range(10):
        idx = np.random.randint(0, nparams)
        e_vec[idx] = e
        loss1 = cofiCostFunc(params - e_vec, Y, myR, nm, nu, nf, l)
        loss2 = cofiCostFunc(params + e_vec, Y, myR, nm, nu, nf, l)
        numgrad = (loss2 - loss1) / (2 * e)
        e_vec[idx] = 0
        diff = np.linalg.norm(numgrad - grad[idx]) / np.linalg.norm(numgrad + grad[idx])
        print('%0.15f \t %0.15f \t %0.15f' % (numgrad, grad[idx], diff))


# 均值归一化
def normalizeRating(Y, R):
    Ymean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1, 1)
    Ynorm = (Y - Ymean) * R
    return Ynorm, Ymean


mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex8\\ex8\\ex8_movies.mat')
Y, R = mat['Y'], mat['R']  # Y为每个用户给每部电影的评分1-5, R为用户是否给电影评分

plt.figure(figsize=(8, 8 * (1682. / 943.)))
plt.imshow(Y, cmap='rainbow')
plt.colorbar()
plt.ylabel('Movies', fontsize=20)
plt.xlabel('User', fontsize=20)
plt.show()

mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex8\\ex8\\ex8_movieParams.mat')
X = mat['X']  # 第i部电影对应的特征向量Xi
Theta = mat['Theta']
nu = int(mat['num_users'])  # 用户数
nm = int(mat['num_movies'])  # 电影数
nf = int(mat['num_features'])  # 特征数
print("nu={},nm={},nf={}".format(nu, nm, nf))

# 测试计算是否正确
'''
nu = 4
nm = 5
nf = 3
X = X[:nm, :nf]
Theta = Theta[:nu, :nf]
Y = Y[:nm, :nu]
R = R[:nm, :nu]
print(cofiCostFunc(serialize(X, Theta), Y, R, nm, nu, nf))
print(cofiCostFunc(serialize(X, Theta), Y, R, nm, nu, nf, 1.5))
print('Checking gradient with lambda = 0...')
checkGradient(serialize(X, Theta), Y, R, nm, nu, nf)
print('Checking gradient with lambda = 1.5...')
checkGradient(serialize(X, Theta), Y, R, nm, nu, nf, 1.5)
'''

# 导入电影数据
movies = []
with open('D:\\Documents\\machine-Leanring\\machine-learning-ex8\\ex8\\movie_ids.txt', 'r', encoding='ISO-8859-1') as f:
    for line in f:
        movies.append(' '.join(line.strip().split(' ')[1:]))

# 我们自己的评分数据
my_ratings = np.zeros((1682, 1))
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

# 看看我们对哪些电影进行了打分
'''
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(my_ratings[i], movies[i])
'''

# 把我们的打分加到原来的数据中
Y = np.c_[Y, my_ratings]
R = np.c_[R, my_ratings != 0]

# 均值归一化数据
Ynorm, Ymean = normalizeRating(Y, R)
nm = Ynorm.shape[0]
nu = Ynorm.shape[1]

# 随机生成特征矩阵X与Theta
X = np.random.random((nm, nf))
Theta = np.random.random((nu, nf))
params = serialize(X, Theta)
l = 10

# 检查梯度下降是否正常工作
# checkGradient(params,Ynorm, R, nm, nu, nf, l)

# 最小化代价函数
res = opt.minimize(fun=cofiCostFunc, x0=params, args=(Ynorm, R, nm, nu, nf, l), method='TNC', jac=cofiGradient,
                   options={'maxiter': 100})
ret = res.x
fit_X, fit_Theta = deserialize(ret, nm, nu, nf)

# 计算预测结果
pred_mat = fit_X @ fit_Theta.T

# 最后一个用户的预测分数， 也就是我们刚才添加的用户
pred = pred_mat[:, -1] + Ymean.flatten()
pred_sorted_idx = np.argsort(pred)[::-1]  # [::-1]从后往前取元素

print("Top recommendations for you:")
for i in range(10):
    print('Predicting rating %0.1f for movie %s.' \
          % (pred[pred_sorted_idx[i]], movies[pred_sorted_idx[i]]))

print("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for movie %s.' % (my_ratings[i], movies[i]))
