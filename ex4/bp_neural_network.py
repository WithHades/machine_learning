import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


# 加载数据
def load_mat(path):
    data = loadmat(path)
    return data['X'], data['y'].flatten()


# 加载权重
def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']


# 展示100张图片
def plot_100_img(X):
    index = np.random.choice(range(5000), 100)
    image = X[index]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(6, 6))
    for row in range(10):
        for col in range(10):
            ax_array[row, col].matshow(image[10 * row + col].reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 将y扩展为向量形式
def expand_y(y):
    result = []

    # 把y中每个类别转化为一个向量，对应的lable值在向量对应位置上置为1
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1
        result.append(y_array)

    # 或者用sklearn中OneHotEncoder函数
    '''
    encoder =  OneHotEncoder(sparse=False)  # return a array instead of matrix
    y_onehot = encoder.fit_transform(y.reshape(-1,1))
    return y_onehot
    '''

    return np.array(result)


# 展开theta值以便能够传入高级优化方法进行运算
def serialize(a, b):
    return np.r_[a.flatten(), b.flatten()]  # np.r_是按行叠加两个矩阵的意思，也可以说是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等


# 还原为原来的theta值
def deserialize(seq):
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 前向传播
def feed_forward(theta, X):
    t1, t2 = deserialize(theta)
    a1 = X

    # 第一层向第二层传播
    z2 = a1 @ t1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)

    # 第二层向第三层传播
    z3 = a2 @ t2.T
    a3 = sigmoid(z3)

    return a1, z2, a2, z3, a3


# 计算代价
def cost(theta, X, y):
    a1, z2, a2, z3, h = feed_forward(theta, X)
    J = 0
    for i in range(len(X)):
        first = - y[i] * np.log(h[i])
        second = (1 - y[i]) * np.log(1 - h[i])
        J = J + np.sum(first - second)
    J = J / len(X)
    return J


# 加入正则的代价函数
def regularized_cost(theta, X, y, l=1):
    t1, t2 = deserialize(theta)
    reg = np.sum(t1[:, 1:] ** 2) + np.sum(t2[:, 1:] ** 2)  # or use np.power(a, 2)
    return l / (2 * len(X)) * reg + cost(theta, X, y)


# 反向传播时S函数的导数
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


# 随机初始化参数, 打破数据的对称性
def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


# 计算梯度
def gradient(theta, X, y):
    t1, t2 = deserialize(theta)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    d3 = h - y  # 最后一层的误差
    d2 = d3 @ t2[:, 1:] * sigmoid_gradient(z2)  # 前一层的误差
    D2 = d3.T @ a2
    D1 = d2.T @ a1
    D = (1 / len(X)) * serialize(D1, D2)
    return D


# 正则的梯度下降
def regularized_gradient(theta, X, y, l=1):
    D1, D2 = deserialize(gradient(theta, X, y))
    t1, t2 = deserialize(theta)
    t1[:, 0] = 0
    t2[:, 0] = 0
    reg_D1 = D1 + (l / len(X)) * t1
    reg_D2 = D2 + (l / len(X)) * t2
    return serialize(reg_D1, reg_D2)


# 在代价函数上沿着切线的方向选择离两个非常近的点然后计算两个点的平均值用以估计梯度。即对于某个特定的theta，我们计算出在theta-e和theta+e的代价的均值
def gradient_checking(theta, X, y, e):
    def a_number_grad(plus, minus):
        return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (e * 2)

    numeric_grad = []
    for i in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] = plus[i] + e
        minus[i] = minus[i] - e
        grad_i = a_number_grad(plus, minus)
        numeric_grad.append(grad_i)

    numeric_grad = np.array(numeric_grad)
    analytic_grad = regularized_gradient(theta, X, y)
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)
    print('If your backpropagation implementation is correct, \nthe relative difference will be smaller than '
          '10e-9.\nRelative Difference:{}\n'.format(diff))


def nn_training(X, y):
    init_theta = random_init(10285)  # 25*401 + 10*26
    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res


def accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))


# 可视化隐藏层
def plot_hidden(theta):
    t1, _ = deserialize(theta)
    t1 = t1[:, 1:]
    fig, ax_array = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6, 6))
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 加载数据
X, raw_y = load_mat('D:\\Documents\\machine-Leanring\\machine-learning-ex4\\ex4\\ex4data1.mat')
# plot_100_img(X)

X = np.insert(X, 0, 1, axis=1)

# y需要扩展为向量形式
y = expand_y(raw_y)

# 加载权重数据
t1, t2 = load_weight('D:\\Documents\\machine-Leanring\\machine-learning-ex4\\ex4\\ex4weights.mat')

# 需要将多个参数矩阵展开，才能传入高级优化方法的优化函数，然后再恢复形状
theta = serialize(t1, t2)
a1, z2, a2, z3, h = feed_forward(theta, X)

# 梯度下降的检验
# gradient_checking(theta, X, y, e=0.0001)

# 训练部分
res = nn_training(X, y)
print(res)

# 计算准确度
accuracy(res.x, X, raw_y)

# 可视化隐藏层
plot_hidden(res.x)
