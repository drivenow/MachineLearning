# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 11:17:29 2016

@author: Administrator
"""
import numpy as np
import csv
from sklearn.metrics import r2_score

'''
    用音乐预测的数据测试s2_score
'''
rootPath = r'D:\OneDrive\codes\matlab\music\RF'
#true = csv.reader(open(rootPath+r'\true', 'r'))
#predict = csv.reader(open(rootPath+r'\predict', 'r'))
true = np.loadtxt(rootPath+r'\true',dtype=int,delimiter=",",skiprows=0)
predict = np.loadtxt(rootPath+r'\predict',dtype=int,delimiter=",")

score = r2_score(true,predict,multioutput= 'uniform_average')

true = np.array([])
for line in true:
    print line
