# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 22:20:01 2016

PCA白化ZCA白化都降低了特征之间相关性较低，同时使得所有特征具有相同的方差。

1.   PCA白化需要保证数据各维度的方差为1，ZCA白化只需保证方差相等。

2.   PCA白化可进行降维也可以去相关性，而ZCA白化主要用于去相关性另外。

3.   ZCA白化相比于PCA白化使得处理后的数据更加的接近原始数据。



@author: Administrator
"""
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.lda import LDA

#rootPath = r'G:\OneDrive\codes\python\RF\data'
rootPath = r'D:\OneDrive\codes\python\RF\data'
trainPath = rootPath+r'\train'
trainlabPath = rootPath+r'\trainlab'

data1 = np.loadtxt(open(trainPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
lab1 = np.loadtxt(open(trainlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)

#%% pca
'''
    PCA
    copy：是否在原数据上进行转换，True,会拷贝一份数据
    whiten:白化,1.特征之间相关性低（SVD分解满足）2.每个特征的方差相同
    method:
    fit:只有监督类,fit方法才有用,fit(X)，表示用数据X来训练PCA模型
    transform:将数据X转换成降维后的数据。
    fit_transform:用X来训练PCA模型，同时返回降维后的数据（直接用transform会失败）
    inverse_transform:将降维后的数据转换成原始数据
'''
pca = PCA(n_components=20, copy=True, whiten=False)
data1 = pca.fit_transform(data1.transpose())#只有监督的转换类才有fit函数，fit(data,lable)


#%% lda
lda = LDA()