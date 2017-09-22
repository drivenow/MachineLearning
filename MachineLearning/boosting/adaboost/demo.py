# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:51:22 2016

@author: Administrator
"""

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit

iris = load_iris()
'''
    adaboost:官网实例，包括本例，有个现象，类似sgd的抖动收敛
    base_estimator:object,(defarult decisionTreeClassifier),弱分类器, Support for sample weighting is required?
    n_estimators:
    learning_rate：shrinks the contribution of each classifier ，和n_estimator此消彼长
    algorithm:(default)'SAMMR.R',real boosting algorithm,base_estimator必须支持对每个类给出预测概率,一般收敛较快
                        'SAMMR':discrete boosting algorithm, 
                        SAMME.R uses the probability estimates to update the additive model, while SAMME uses the classifications only
    
    $Attributs:
    classes:calss label
    n_classes:class num
    estimator_error:error for each estimator in boosted ensemble
    estimator_weight:
    
'''
clf = AdaBoostClassifier(DecisionTreeClassifier(max_features='sqrt'),n_estimators=300,learning_rate=0.1)
ssp = StratifiedShuffleSplit(iris.target,n_iter=3,test_size=0.9,random_state=1)
scores = cross_val_score(clf, iris.data, iris.target,cv=ssp)
print(scores.mean())                             
