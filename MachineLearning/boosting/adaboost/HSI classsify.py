# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:49:14 2016

@author: Administrator
"""
'''
    adaboost方法，选择树分类器，效果没有随机森林好
    选择逻辑回归分类器，
'''
import numpy as np 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import treeinterpreter.treeinterpreter as ti
import gc
import os
from sklearn.decomposition import PCA
from sklearn.pipeline import  Pipeline
from time import time

#%%
#load data
rootPath = r'G:\OneDrive\codes\python\RF\data'
#rootPath = r'D:\OneDrive\codes\python\RF\data'
trainPath = rootPath+r'\train'
#trainPath = rootPath+r'\train_filter'
trainlabPath = rootPath+r'\trainlab'
#testPath = rootPath+r'\test'
testPath = rootPath+r'\test'
testlabPath = rootPath+r'\testlab'

data1 = np.loadtxt(open(trainPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
lab1 = np.loadtxt(open(trainlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
data2 = np.loadtxt(open(testPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
lab2 = np.loadtxt(open(testlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
test = {'data':data2.transpose(),'lab':lab2}
train = {'data':data1.transpose(),'lab':lab1}


#ssp = StratifiedShuffleSplit(test['lab'],n_iter=1,test_size=0.9,random_state=1)
ssp = StratifiedShuffleSplit(train['lab'],n_iter=3,test_size=0.1,random_state=1)
for trainlab,testlab in ssp:
    print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
    X = test['data'][trainlab];Xlab = test['lab'][trainlab]
    Y = test['data'][testlab];Ylab = test['lab'][testlab]


#%% pipeliene&cv
pca = PCA()
adaboost_tree = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_features='auto',criterion="entropy"))
adaboost_log = AdaBoostClassifier(base_estimator=LogisticRegression(C=1000,class_weight=None,penalty='l2'
        ,solver='newton-cg',max_iter=200,tol=1E-5,n_jobs=1,verbose=3,
        multi_class='multinomial', fit_intercept=True))

#%% 选择模型
"""
 By default, weak learners are decision stumps.默认的基分类器是决策树
learning_rate:controls the contribution of the weak learners in the final combination. 
main parameters to tune：n_estimators and the complexity of the base estimators 
"""
#pipeline = Pipeline([('pca',pca),('ada',adaboost_tree)])
pipeline = Pipeline([('pca',pca),('ada',adaboost_log)])
#paramMap = {'ada__n_estimators':[175,150,200],'ada__learning_rate':[1E-3,1E-1,1],'ada__algorithm':["SAMME",'SAMME.R'],
#            'pca__n_components':[200,50,25,16],"pca__whiten":[True]}
paramMap = {'ada__n_estimators':[50],'ada__learning_rate':[1],'ada__algorithm':["SAMME",'SAMME.R'],
            'pca__n_components':[25],"pca__whiten":[True]}
            
#cv 选择一个cv生成器
#refit =False,表示不用整个数据集来refit,best_estimator
#grid_search = GridSearchCV(pipeline,paramMap,refit=False,n_jobs=1,cv=ssp)
grid_search = GridSearchCV(pipeline,paramMap,refit=True,n_jobs=3,cv=ssp)
#%% 
# training
print("start training···")
start = time()
#grid_search.fit(test['data'],test['lab'])
grid_search.fit(train['data'],train['lab'])
print("GridSearchCV took %.2f seconds for %d cadidate param sets"
      %(time()-start,len(grid_search.grid_scores_)))

best_score= grid_search.best_score_
best_param = grid_search.best_params_
#%%
#testing
if os.path.exists(rootPath+'\prediction'):
    os.remove(rootPath+'\prediction')#删除文件
row,col = test['data'].shape
prediction = np.array([])
for i in range(int(np.ceil(row/1000.0))):
    start = 1000*i
#    prediction,bias,contributions, =ti.predict(rf,test['data'][start:min(start+20,col)])
    prediction = np.concatenate((prediction,grid_search.predict(test['data'][start:min(start+1000,row)])),axis=1)

#追加到文件
np.savetxt(open(rootPath+'\prediction','w'),prediction,delimiter = ',',fmt ='%d')
    
print("calssification report:")
report = classification_report(test['lab'],prediction)
acu=accuracy_score(test['lab'],prediction)


