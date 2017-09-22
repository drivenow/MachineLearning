# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:27:46 2016

@author: Shenjunling

66%准确率
"""


import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import os
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.pipeline import  Pipeline
from time import time


#%%
#load data
rootPath = r'G:\OneDrive\codes\python\RF\data'
#rootPath = r'D:\OneDrive\codes\python\RF\data'
#trainPath = rootPath+r'\train_filter'
trainPath = rootPath+r'\train'
trainlabPath = rootPath+r'\trainlab'
trainidxPath = rootPath+r'\trainidx'
testPath = rootPath+r'\test'
#testPath = rootPath+r'\test_filter'
testlabPath = rootPath+r'\testlab'
testidxPath = rootPath+r'\testidx'
imgPath = rootPath+r'\img'

data1 = np.loadtxt(open(trainPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
lab1 = np.loadtxt(open(trainlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
idx1 = np.loadtxt(open(trainidxPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
data2 = np.loadtxt(open(testPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
lab2 = np.loadtxt(open(testlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
idx2 = np.loadtxt(open(testidxPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
test = {'data':data2.transpose(),'lab':lab2,'idx':idx2}
train = {'data':data1.transpose(),'lab':lab1,'idx':idx1}
img = np.loadtxt(open(imgPath,"rb"),delimiter=',',skiprows=0,dtype=np.float)
img = img.transpose()

#ssp = StratifiedShuffleSplit(test['lab'],n_iter=1,test_size=0.9,random_state=1)
ssp = StratifiedShuffleSplit(train['lab'],n_iter=4,test_size=0.1,random_state=1)
for trainlab,testlab in ssp:
    print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
    X = test['data'][trainlab];Xlab = test['lab'][trainlab]
    Y = test['data'][testlab];Ylab = test['lab'][testlab]



#%% pipeliene&cv
"""
   LLE:流形学习方法
   method: method = 'modified'. It requires n_neighbors > n_components.
                    'hessian', n_neighbors > [n_components * (n_components + 3) / 2]
 
   eighn_solver: ['auto,'arpack','dense']auto : 自动选择最佳方法，但有时选到‘arpack’会报错
                   ‘arcapk’:XtMX中的M可以是稀疏矩阵或者线性算子，但是求解不稳定，有时需要改变随机种子
                   ‘dense’:M是dense矩阵，避免大矩阵问题使用
"""
lle = LocallyLinearEmbedding()
i = 1
if i==1:rf = RandomForestClassifier(n_jobs=3,oob_score=True,bootstrap=True,min_samples_split=2,min_samples_leaf=1)
elif i==2:rf = ExtraTreesClassifier(n_jobs=3,oob_score=True,bootstrap=True)


#%% 选择模型
#pipeline = Pipeline([('pca',pca),('rf',rf)])
pipeline = Pipeline([('lle',lle),('rf',rf)])
paramMap = {'rf__n_estimators':[300],'rf__max_features':['auto'],'rf__criterion':["entropy"],
            'lle__n_neighbors':[20],'lle__n_components':[5,10,20],'lle__method':['standard','ltsa'],
             'lle__eigen_solver':['dense'],'lle__neighbors_algorithm':['auto','brute','kd_tree','ball_tree']}
#cv 选择一个cv生成器
#refit =False,表示不用整个数据集来refit,best_estimator
#grid_search = GridSearchCV(pipeline,paramMap,refit=False,n_jobs=1,cv=ssp)
grid_search = GridSearchCV(pipeline,paramMap,refit=True,n_jobs=1,cv=ssp)            
            
#%% 
# training
print("start training···")
start = time()
#grid_search.fit(test['data'],test['lab'])
#grid_search.fit(train['data'],train['lab'])
grid_search.fit(img[train['idx'],:],train['lab'])
print("GridSearchCV took %.2f seconds for %d cadidate param sets"
      %(time()-start,len(grid_search.grid_scores_)))

best_score= grid_search.best_score_
best_param = grid_search.best_params_
#%%
#testing
#grid_search 缺少best_estimator，是因为refit参数是false
if os.path.exists(rootPath+'\prediction'):
    os.remove(rootPath+'\prediction')#删除文件
#row,col = test['data'].shape
row,col = img.shape
prediction = np.array([])
for i in range(int(np.ceil(row/1000.0))):
    
    start = 1000*i
    if i==0:
        prediction = grid_search.predict_proba(img[start:min(start+1000,row)])
    else:
#    prediction,bias,contributions, =ti.predict(rf,test['data'][start:min(start+20,col)])
    #合并predict向量
#    prediction = np.concatenate((prediction,grid_search.predict(test['data'][start:min(start+1000,row)])),axis=1)
    #合并prob矩阵
        prediction = np.concatenate((prediction,grid_search.predict_proba(img[start:min(start+1000,row)])),axis=0)

#追加到文件
np.savetxt(open(rootPath+'\prediction','w'),prediction,delimiter = ',',fmt ='%f')

print("calssification report:")
prediction_i = np.zeros(prediction.shape[0],dtype=np.int)
for i in range(prediction.shape[0]):
    idx=0
    max = prediction[i][0]
    for j in range(16):
        if max < prediction[i][j]:
            max = prediction[i][j] 
            idx =j+1
    prediction_i[i]=np.int(idx)
prediction_ii = prediction_i[test['idx']-1]
report = classification_report(test['lab'],prediction_ii)
#g = open(rootPath+'\classification report','a')
#g.write(report)

