# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:27:46 2016
高光谱图像分类：用滤波器预处理
@author: Shenjunling
"""

import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import  Pipeline
from time import time
from scipy.ndimage import filters
import sys
sys.path.append("D:/OneDrive/codes/python/DataLoad")
from HSIDataLoad import *

#%% dataset3
#dataset3数据集，滤波前73，滤波后86，用自己写的高斯滤波卷积
#dataset2数据集，滤波前76，滤波后87

#%% dataset2
rootPath = "../data/HSI"
X_data,Y_data,data_source,idx_data=datasetLoad2(rootPath)#未划分训练集测试集的数据(不包括背景点)
Y_data=np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(X_data,Y_data,idx_data,num_calss=16,test_size=0.9)
Y_train=np_utils.categorical_probas_to_classes(Y_train)+1
Y_test=np_utils.categorical_probas_to_classes(Y_test)+1

#%% args
rf_choose = 2 #1 for RandomForestClassifier,2 for ExtraTreesClassifier
paramMap = {'rf__n_estimators':[300],'rf__max_features':['auto'],'rf__criterion':["entropy"],
            'rf__class_weight':['balanced']}
ssp = StratifiedShuffleSplit(Y_train,n_iter=2,test_size=0.1,random_state=1)
grid_search_refit=True#在grid_search中用最佳参数，重训练模型
g_filter=True#使用高斯滤波


#%% gaussian filter
import copy
import itertools

def guass_dist(mu1, mu2, sig, rho, window):
     sig2 = sig**2
     kernel = np.zeros(window**2)
     for i in range(window):
         for j in range(window):
             e = ((i-mu1)**2+(j-mu2)**2)/2/sig2
             f = 1/(2*3.14*sig2)*np.exp(-1/(2*sig2)*e)
             kernel[i*window+j] = f
     return kernel
"""
mode0: head padding
mode1: tail padding
""" 
def padding(lista, length, mode):
    diff = length-len(lista)
    if mode==0:
        pad = [0]*diff
        pad.extend(lista)
        return pad
    if mode==1:
        pad = [144]*diff
        lista.extend(pad)
        return lista
"""
对图像卷积
"""   
def conv(img,kernel):
    row,col = img.shape
    img_copy = copy.deepcopy(img)
    window = int(np.sqrt(len(kernel)))
    wei = int((window-1)/2)
    for i in range(row):
        for j in range(col):
            x_range = list(range(max(0,i-wei),min(145,i+wei+1)))
            y_range = list(range(max(0,j-wei),min(145,j+wei+1)))
            if i < wei:
                x_range = padding(x_range, window, 0)
            if i > row-1-wei:
                x_range = padding(x_range, window, 1)
            if j < wei:
                y_range = padding(y_range, window, 0)
            if j > col-1-wei:
                y_range = padding(y_range, window, 1)
            pixel_window=[]
            for pair in itertools.product(x_range,y_range):
                pixel_window.append(img_copy[pair])
            pixel_window = np.array(pixel_window)
            img[i,j] = np.sum(pixel_window*kernel)
    return img

     
my_filter = guass_dist(1,1,1,0,3)
#my_filter = guass_dist(3,3,1,0,7)
if g_filter==True:
    img_temp = []
    for i in range(200):
        img_temp = data_source[:,i]
        img_2d = np.reshape(img_temp,(145,145))
#        img_2d = filters.gaussian_filter(img_2d,0.34)#卷积窗口https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
        img_2d = conv(img_2d, my_filter)   
        data_source[:,i] = np.reshape(img_2d,21025)

#%% 选择模型
pca = PCA()
if rf_choose==1:rf = RandomForestClassifier(n_jobs=3,oob_score=True,bootstrap=True,min_samples_split=2,min_samples_leaf=1)
elif rf_choose==2:rf = ExtraTreesClassifier(n_jobs=3,oob_score=True,bootstrap=True)

pipeline = Pipeline([('rf',rf)])
#cv 选择一个cv生成器
#refit =False,表示不用整个数据集来refit,best_estimator
#grid_search 缺少best_estimator，是因为refit参数是false
grid_search = GridSearchCV(pipeline,paramMap,refit=grid_search_refit,n_jobs=1,cv=ssp) 
           
            
#%% training
print("start training···")
start = time()
grid_search.fit(data_source[idx_train],Y_train)
print("GridSearchCV took %.2f seconds for %d cadidate param sets"
      %(time()-start,len(grid_search.grid_scores_)))

best_score= grid_search.best_score_
best_param = grid_search.best_params_


#%% testing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#判断f1和总的准确率，输入的Y_test是类别形式
def modelMetricsCtg(model_fitted,X_test,Y_test):
    Y_predict_ctg=model_fitted.predict(X_test)
    report =classification_report(Y_predict_ctg,Y_test)#各个类的f1score
    accuracy = accuracy_score(Y_predict_ctg,Y_test)#总的准确度
    return report,accuracy

report,accuracy=modelMetricsCtg(grid_search,data_source[idx_test],Y_test)





