# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:27:46 2016

@author: Shenjunling
"""

import numpy as np 
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time


def getData(rootPath):
    imgPath = rootPath+r'\labeled_data.2.28.txt'
    labPath = rootPath+r'\data_label.2.28.txt'
 
    data = np.loadtxt(open(imgPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    lab = np.loadtxt(open(labPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)

    return (data,lab)
    
def getCV(lab, test_size):
    #划分训练集和验证集
    ssp = StratifiedShuffleSplit(lab,n_iter=1,test_size=test_size,random_state=1)
    X = {}
    Y = {}#训练集
    for trainlab,testlab in ssp:
        print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
        X['data'] = data[trainlab]; X['lab'] = lab[trainlab]
        Y['data'] = data[testlab]; Y['lab'] = lab[testlab]
    return X,Y

'''
    copy：是否在原数据上进行转换，True,会拷贝一份数据
    whiten:白化
    method:
    fit:只有监督类,fit方法才有用,fit(X)，表示用数据X来训练PCA模型
    transform:将数据X转换成降维后的数据。
    fit_transform:用X来训练PCA模型，同时返回降维后的数据（直接用transform会失败）
    inverse_transform:将降维后的数据转换成原始数据
'''
def getModel_RF(i, ssp):
    pca = PCA()
    if i==1:rf = RandomForestClassifier(n_jobs=3,oob_score=True,bootstrap=True,min_samples_split=2,min_samples_leaf=1)
    elif i==2:rf = ExtraTreesClassifier(n_jobs=3,oob_score=True,bootstrap=True)
    
    pipeline = Pipeline([('pca',pca),('rf',rf)])
    paramMap = {'rf__n_estimators':[300],'rf__max_features':['auto','log2'],'rf__criterion':["entropy","gini"],
                'rf__class_weight':['balanced',None],'pca__n_components':[50],"pca__whiten":[True,False]}
    #cv 选择一个cv生成器
    grid_search = GridSearchCV(pipeline,paramMap,refit=True,n_jobs=1,cv=ssp)
    return grid_search
    
    
if __name__=="__main__":
    rootPath = "..\data\HSI"
    data,lab = getData(rootPath)
    train,test = getCV(lab, test_size=0.9)
    
    ssp = StratifiedShuffleSplit(train['lab'],n_iter=1,test_size=0.9,random_state=1)
    model_rf = getModel_RF(1,ssp)
    
    #train model
    print("start training...")
    start = time()
    model_rf.fit(train['data'],train['lab'])
    print("GridSearchCV took %.2f seconds for %d cadidate param sets"
          %(time()-start,len(model_rf.grid_scores_)))
    
    #test model
    best_score= model_rf.best_score_
    best_param = model_rf.best_params_
    
    prediction = model_rf.predict(test['data'])
    report = classification_report(test['lab'],prediction)
    accu = accuracy_score(test['lab'],prediction)
    print(accu)