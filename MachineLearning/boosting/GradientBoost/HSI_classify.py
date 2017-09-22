# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:09:00 2016
@author: Shenjunling
"""

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from xgboost.sklearn import XGBClassifier
import numpy as np
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

    
def getModel_GBDT(ssp):
    model = GradientBoostingClassifier(random_state=11111)
    pipeline = Pipeline([('gb',model)])
    paramMap = {'gb__n_estimators':[500],'gb__learning_rate':[1E-1],
                'gb__loss':['deviance'],'gb__max_depth':[5],'gb__subsample':[0.8],
                'gb__max_features':['auto']}
    #cv 选择一个cv生成器
    #refit =False,表示不用整个数据集来refit,best_estimator
    grid_search = GridSearchCV(pipeline,paramMap,refit=True,n_jobs=3,cv=ssp)
    return grid_search
    
def getModel_RF(ssp):
    rf = RandomForestClassifier(n_jobs=1,oob_score=True,bootstrap=True,min_samples_split=2,min_samples_leaf=1)
    
    pipeline = Pipeline([('rf',rf)])
    paramMap = {'rf__n_estimators':[500],'rf__max_features':['auto','log2'],'rf__criterion':["entropy","gini"],
                'rf__class_weight':['balanced',None]}
    #cv 选择一个cv生成器
    grid_search = GridSearchCV(pipeline,paramMap,refit=True,n_jobs=1,cv=ssp)
    return grid_search
    

def getModel_XGB(ssp, early_stopping_rounds=50):
    model = XGBClassifier(learning_rate =0.1,n_estimators=500,
         max_depth=5, min_child_weight=1, gamma=0,
         subsample=0.8,colsample_bytree=0.8,
         objective= 'multi:softmax',  nthread=2,
         scale_pos_weight=1,seed=27)
    
    paramMap = {'n_estimators':[300],'learning_rate':[1E-1],'max_depth':[5],
                'gamma':[1], 'subsample':[0.8]}
    #cv 选择一个cv生成器
    model1 = GridSearchCV(model,paramMap,refit=True,n_jobs=1,cv=ssp)
             
    return model1
    
#    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#    feat_imp.plot(kind='bar', title='Feature Importances')
#    plt.ylabel('Feature Importance Score')

    
def testModel(model, data ,lab):
    #test model
    best_score= model.best_score_
    best_param = model.best_params_
    
    prediction = model.predict(data)
    report = classification_report(lab,prediction)
    test_accu = accuracy_score(lab,prediction)
    
    return best_score,best_param, report, test_accu
    
def testModel1(model, data ,lab):

    prediction = model.predict(data)
    report = classification_report(lab,prediction)
    test_accu = accuracy_score(lab,prediction)
    
    return report, test_accu
    
    
    
if __name__=="__main__":
    rootPath = r'..\..\data\HSI'
    data,lab = getData(rootPath)
    #划分训练集和测试集，ssp是迭代器
    X,Y = getCV(lab,test_size=0.9)
    
    ssp = StratifiedShuffleSplit(X['lab'],n_iter=1,test_size=0.9,random_state=1)
    model_gbdt = getModel_GBDT(ssp)
    model_rf = getModel_RF(ssp)
    model_xgb = getModel_XGB(ssp)

    
    switch = [0,2]#选择一个分类器
    models = []
    models.append(model_gbdt)
    models.append(model_rf)
    models.append(model_xgb)
    
    for i in switch:
        #train model
        print("start training...")
        start = time()
        if i<2:
            models[i].fit(X['data'],X['lab'])
            print("model %d GridSearchCV took %.2f seconds for %d cadidate param sets"
              %(i+1,time()-start,len(models[i].grid_scores_)))
        
            #test model
            best_score,best_param,report,accu = testModel(models[i], Y['data'], Y['lab'])
            print(accu)
        else:
            models[i].fit(X['data'],X['lab']-1)
            print("model %d GridSearchCV took %.2f seconds for %d cadidate param sets"
              %(i+1,time()-start,len(models[i].grid_scores_)))
        
            #test model
            best_score,best_param,report,accu = testModel(models[i], Y['data'], Y['lab']-1)
            print(accu)
    



