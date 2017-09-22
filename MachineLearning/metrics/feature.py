# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 20:38:36 2016

@author: Administrator
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import scipy as sp 
from numpy.linalg import norm

iris = load_iris()

# 对每一个sample,各个特征加权x[i]/norm(x,2),i=0,1``
iris_norm = Normalizer(norm='l2').fit_transform(iris['data'])
# 多项式特征
"""
   可实现非线性拟合，也是一种核化方法
   degree:二项式最高项,比如（x1,x2）的2次多项式，1,x1,x2,x1*x1,x1*x2,x2*x2
   interaction_only:只允许交互，即不允许出现x1*X1,
"""
iris_poly = PolynomialFeatures()

a =  iris['data'][0]
sum = 0 
for i in range(len(a)):
    sum+=a[i]**2
    
a[0]/np.sqrt(sum)

norm