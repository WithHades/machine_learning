import re

import nltk
import nltk.stem.porter
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import svm


# 预处理邮件, 所有字符小写并且替换所有html标签/网页链接/邮箱地址/美元符号/数字
def processEmail(email):
    email = email.lower()
    email = re.sub('<[^<>]>', '', email)
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub('[$]+', 'dollar', email)
    email = re.sub('[0-9]+', 'number', email)
    return email


# 邮件转为单词列表
def email2TokenList(email):
    stemmer = nltk.stem.porter.PorterStemmer()
    email = processEmail(email)

    # 通过正则指定多种分隔符, 实现字符分割
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    tokenlist = []
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)  # 删除非字母数字字符
        stemmed = stemmer.stem(token)  # 数据正规化,即全部转为原始形式,比如,depends变为depend
        if not len(stemmed): continue  # 空则跳过
        tokenlist.append(stemmed)
    return tokenlist


# 将邮件根据单词表转为向量形式
def email2FeatureVector(email):
    token = email2TokenList(email)
    vector = [1 if vocab[i] in token else 0 for i in range(len(vocab))]
    return np.array(vector)


# 读取邮件
with open('D:\\Documents\\machine-Leanring\\machine-learning-ex6\\ex6\\emailSample1.txt', 'r') as f:
    email = f.read()
    print(email)

# 读取单词表
vocab = pd.read_csv('D:\\Documents\\machine-Leanring\\machine-learning-ex6\\ex6\\vocab.txt', names=['words'], sep='\t').values
vector = email2FeatureVector(email)
print('length of vector = {}\nnum of non-zero = {}'.format(len(vector), int(vector.sum())))

# 以上过程演示了如何将邮件向量化,可以根据上述步骤对大量邮件预处理作为输入数据

# 读取已经预处理好的邮件以及相应的标签,分为训练集和测试集

mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex6\\ex6\\spamTrain.mat')
X, y = mat['X'], mat['y']

mat = loadmat('D:\\Documents\\machine-Leanring\\machine-learning-ex6\\ex6\\spamTest.mat')
Xtest, ytest = mat['Xtest'], mat['ytest']

# 采用SVM训练
clf = svm.SVC(C=0.1, kernel='linear')
clf.fit(X, y.flatten())

# 检查训练效果
predTrain = clf.score(X, y)
predTest = clf.score(Xtest, ytest)
print('PredTrain={}, PredTest={}'.format(predTrain,predTest))

# 可以接着对C的值进行调整或者采用高斯内核尝试分类, 由于精度很高,不再进行尝试
