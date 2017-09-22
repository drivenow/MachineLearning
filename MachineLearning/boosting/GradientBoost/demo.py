# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:50:09 2016

@author: Shenjunling
"""

#Import Library
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import minmax_scale
import sklearn.datasets
import numpy as np

iris = sklearn.datasets.load_iris()
data = minmax_scale(iris.data)#对每一列归一化
target = iris.target
#print iris.DESCR

idx = range(len(iris.target))
np.random.shuffle(idx)

#%%
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
'''
    gradient boost classify:收敛性不错
    The advantages of GBRT are:
            Natural handling of data of mixed type (= heterogeneous features)
            Predictive power
            Robustness to outliers in output space (via robust loss functions)
    The disadvantages of GBRT are:
            Scalability, due to the sequential nature of boosting it can hardly be parallelized.
    n_estimator:与RF不同，对多分类问题，每个类有n_estimator棵数，总共N_estimatot*nclass棵
    loss:(default)deviance(异常)：expomential(指数)：recover adaboost算法
    subample:(default 1.0),bootstrap avaraging(bagging) 采样的比例,随机森林中feature采样也是这种方法
'''
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# Train the model using the training sets and check score
model.fit(data[idx[:100]], target[idx[:100]])

#Predict Output
predicted= model.predict(data[idx[100:150]])

#score
score = model.score(data[idx[100:150]],target[idx[100:150]])
ssp = StratifiedShuffleSplit(iris.target,n_iter=3,test_size=0.9)
scores = cross_val_score(model, iris.data, iris.target,cv=ssp)
scores1 = cross_val_score(model, data, target,cv=ssp)
print(scores.mean())
print(scores1.mean())