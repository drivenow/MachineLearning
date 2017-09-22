# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:02:41 2016

@author: Administrator

比较GridSearchCV和RandomizedSearchCV在挑选最优参数上的时间效率
"""

print(__doc__)

import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# get some data
digits = load_digits()
X, y = digits.data, digits.target

# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# specify parameters and distributions to sample from
'''RandomizedSearchCV
    param_distributions:distributions or lists. Or If a list is given, it is sampled uniformly.
    sp_randint(1, 11):
'''
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
'''
    RandomizedSearchCV:对参数抽样进行试验
    n_iter:参数采样的次数
'''
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
'''
    exhaustive grid_search:对一组参数集合寻找最佳参数
    refit=Ture:是否在得到best_estimator之后，再用整个数据集进行refit
    n_jobs:并行运行的数目
    verbose:整数，越大打印信息越详细
    scoring: 1.string ,2.a scorer callable object function with signature scorer(estimator, X, y).   
    cv： int, cross-validation generator or an iterable, optional.default(3)如果是整数，交叉验证集一共有多少组(默认共享一组shuffle的数据)。如果是多分类问题StratifiedKFold，如果不是分类问题，KFolds
'''

grid_search = GridSearchCV(clf, param_grid=param_grid，refit=False)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)