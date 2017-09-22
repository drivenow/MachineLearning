# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 21:55:01 2016

@author: Administrator
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm


'''''数据生成'''
x = np.arange(0,1,0.002)
y =  norm.rvs(0,size=500,scale=0.1)
y = y+x**2

'''''均方根误差'''
def rmse(y_test,y):
    return sp.sqrt(sp.mean((y_test-y)**2)) #(b-a)**2  array([1, 1])
    
'''''与均值相比的·1优秀程度，0表示不如均值，1表示完美预测'''
def R2(y_test,y):
    return 1-((y_test-y)**2).sum()/((y_test-y.mean())**2).sum()

'''''参数item__param，如poly__degree'''
plt.scatter(x,y,s=5)
paramMap=dict(poly__degree=[1,2,100])

'''single model'''
poly = PolynomialFeatures() #对X进行多项式变换,(2,3),drgree2->(1,2,3,4,9,6)
lr = LinearRegression()

''''' pipeline'''
pipeline = Pipeline([('poly',poly),('lr',lr)])
pipeline.set_params(poly__degree=2)

#%% grid_search
'''
    exhaustive grid_search:对一组参数集合寻找最佳参数
    refit=Ture:是否在得到best_estimator之后，再用整个数据集进行refit。注意只有refit为true,才能用grid_search的predict函数
    n_jobs:并行运行的数目
    verbose:整数，越大打印信息越详细
    scoring: 1.string ,2.a scorer callable object function with signature scorer(estimator, X, y).   
    cv： int, cross-validation generator or an iterable, optional.default(3)如果是整数，交叉验证集一共有多少组(默认共享一组shuffle的数据)。如果是多分类问题StratifiedKFold，如果不是分类问题，KFolds
'''
grid_search = GridSearchCV(pipeline,paramMap,n_jobs=3,cv=2,verbose=2)
print("Performing grid search...")  
print("pipeline:", [name for name, _ in pipeline.steps])  
print("parameters:")  
print(paramMap)  
#train
grid_search.fit(x[:,np.newaxis],y)
#best param
grid_search.best_estimator_.named_steps['poly']#PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
grid_search.best_estimator_.named_steps['poly'].degree
grid_search.best_params_#{'poly__degree': 2}
grid_search.best_score_
#predict,默认使用最佳参数进行predict
y_test = grid_search.predict(x[:,np.newaxis])

plt.plot(x,y_test,linewidth =2)
plt.legend(['1','2','100'], loc='upper left') 
plt.show()  

'''
    RandomizedSearchCV:对参数抽样进行试验
    n_iter:参数采样的次数
'''

#%%
'''crossvalidation'''
iris = datasets.load_iris()
sample,feature = iris.data.shape
clf = svm.SVC(kernel='linear',C=1)
'''
    train_test_split:按照比例，随机划分数据集(始终随机)
    缺点：不能按照类别，等比例的抽样
'''
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(iris.data,
iris.target,test_size=0.4,random_state=0)#eandom_state：伪随机数种子，设定其有助于重现划分
'''
    KFold：返回的是一个划分，比如十个样本1~10，分成两组test和train,[1-5],[6-10]各为一个测试集，分成三组，[1,2,3],[4,5,6],[7,8,,9,10]各为一个测试集
    shuffle : default False,是否在划分前打乱数据
    random_state :shuffle为True的情况下，设置伪随机数种子
'''
kf = cross_validation.KFold(sample,3)
for trainidx,testidx in kf:
    print("%s %s" % (trainidx, testidx))
    X_train, X_test, y_train, y_test = iris.data[trainidx], iris.data[testidx], iris.target[trainidx], iris.target[testidx]
'''
    Stratified KFold:分层，根据lable的值选取数据，可以保证每类样本都抽取一定比例
    同KFold:包含shuffle,random_number
    不同KFol:n不是样本数目，而是array[lable]
'''
skf = cross_validation.StratifiedKFold(iris.target,3)
for trainidx,testidx in skf:
    print("%s %s" % (trainidx,testidx))
    train = iris.target[trainidx]
    print("calss 1 proportion %.3f" % ((train[train==1].sum())/float(iris.target[iris.target==1].sum())))
    print("calss 2 proportion %.3f" % (train[train==2].sum()/float(iris.target[iris.target==2].sum())))
'''
    Lable KFold:train和test中没有相同的类，应用，识别某个人的特征，最后一个人作为测试集，它不应该出现在训练集中
'''
labels = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
lkf = cross_validation.LabelKFold(labels, n_folds=3)
for train, test in lkf:
    print("%s %s" % (train, test))
'''
    留下固定的样本数
     LeaveOneOut:
     LeavePOut(4, p=2)
     LeaveOneLabelOut(labels)
     LeavePLabelOut(labels, p=2)：训练集和测试集不存在交叉的lable
'''
'''
    shuffelsplit:是Kfold的另一种实现方式，
    n_iter:多少次shuffel,一次产生一组数据
    test_size:每次测试集多占的比例
    random_state:伪随机数种子
    1.cross_validation.ShuffleSplit(5, n_iter=3, test_size=0.25,random_state=0)
    2.LabelShuffleSplit(labels, n_iter=4, test_size=0.5,random_state=0)
    3.StratifiedShuffleSplit(lables, 3, test_size=0.5, random_state=0)
'''

#%% scoring    
'''
    cross_val_score:训练加预测，返回多次交叉验证数据集上的score
    cv:cv：交叉验证集一共有多少组(所有参数集，都共享一组shuffle的数据)。
        如果是多(二)分类问题StratifiedKFold;如果不是分类问题，KFolds
        defalut,None, to use the default 3-fold cross-validation
    fit_params：字典，传递给estimator的参数
    
'''
scores = cross_validation.cross_val_score(
    clf,iris.data,iris.target,cv=5)



