# -*- coding: utf-8 -*-
"""
Created on Sun May 14 09:36:57 2017

@author: Shenjunling
"""
import numpy as np

def loadExData2():
    return np.array([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

a = loadExData2()
"""
（1）用特征值分解求pca
"""
#零均值化  
def zeroMean(dataMat):        
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值  
    newData=dataMat-meanVal  
    return newData,meanVal  
  
def pca(dataMat,n):  
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本  
      
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量  
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标  
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量  
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据  
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据  
    return lowDDataMat,reconMat  
    
pca0 = pca(a,7)
pca3 = np.array(pca0[0])



"""
(2) 用svd实现pca降维
提供单个方向的的PCA(纵坐标降维)
"""

from sklearn.decomposition import PCA
pca_model = PCA(n_components=7, svd_solver="full")
pca1 = pca_model.fit_transform(a)


"""
使用svd进行PCA分解,注意：输入数据是减去横坐标均值的
U,S,V = svd(x),S是特征值的根号值，V是特征向量的转秩
U.T*x = S*V
x*V.T = U*V
"""
a_mean = np.mean(a,axis=0)
aa = a-a_mean
U,S,V = np.linalg.svd(aa)
Ua = np.array(U)
pca2 = np.dot(Ua[:,:7], np.diag(S)[:7,:7])


"""
svd压缩图像（保留长宽，但降秩）
x = U[:,:n]*np.diag(S[:n]*V[:n,n])
"""
