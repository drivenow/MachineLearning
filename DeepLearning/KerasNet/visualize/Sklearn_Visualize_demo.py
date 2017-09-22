# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:41:29 2017

@author: Administrator
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
iris = load_iris()#载入数据集
clf = tree.DecisionTreeClassifier()#算法模型
clf = clf.fit(iris.data, iris.target)#模型训练
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph=graph[0]
graph.write_pdf("output/iris.pdf")#写入pdf