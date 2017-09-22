# -*- coding: utf-8 -*-



import numpy as np 
import sklearn.decomposition
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import neighbors

#%%
#load data
#rootPath = r'G:\OneDrive\codes\python\RF\data'
rootPath = r'D:\OneDrive\codes\python\RF\data'
trainPath = rootPath+r'\train'
trainlabPath = rootPath+r'\trainlab'
testPath = rootPath+r'\test'
testlabPath = rootPath+r'\testlab'

data1 = np.loadtxt(open(trainPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
lab1 = np.loadtxt(open(trainlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
data2 = np.loadtxt(open(testPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
lab2 = np.loadtxt(open(testlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)

train = {'data':data1.transpose(),'lab':lab1}
test = {'data':data2.transpose(),'lab':lab2}


#%% 数据预处理,
'''
    PCA降维之后效果变差
    copy：是否在原数据上进行转换，True,会拷贝一份数据
    whiten:白化
    method:
    fit:只有监督类,fit方法才有用,fit(X)，表示用数据X来训练PCA模型
    transform:将数据X转换成降维后的数据。
    fit_transform:用X来训练PCA模型，同时返回降维后的数据（直接用transform会失败）
    inverse_transform:将降维后的数据转换成原始数据
'''
#pca = sklearn.decomposition.PCA(n_components=20, copy=True, whiten=False)
#data1 = pca.fit_transform(data1.transpose())#只有监督的转换类才有fit函数，fit(data,lable)
#data1 = data1.transpose()
#data2 = pca.fit_transform(data2.transpose())
#data2 = data2.transpose()


#%% 
# training
"""
weight:'uniform'近邻的每个样本的权重是统一的， 'distance'，近邻权重跟距离成反比
metric：default = ‘minkowski’距离函数，也可以是欧式距离等
metric_params：metrics的参数
p:Power parameter for the Minkowski metric. ,默认2

"""
n_neighbors = 10
acu=0
for algo in ['auto', 'ball_tree', 'kd_tree', 'brute']:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform',algorithm=algo)
    clf.fit(train['data'], train['lab'])
    Z = clf.predict(test['data'])
    report =classification_report(Z,test['lab'])
    acu=accuracy_score(Z,test['lab'])
    if algo=='auto':
        tmp=acu
        tmp_report=report
    if acu>tmp:
        tmp=acu
        tmp_report=report
        arg=algo
        






