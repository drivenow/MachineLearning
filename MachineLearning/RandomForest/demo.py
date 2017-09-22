# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:16:40 2016


criterion  ：规定了该决策树所采用的的最佳分割属性的判决方法，有两种：“gini”，“entropy”。

max_depth  ：限定了决策树的最大深度，对于防止过拟合非常有用。

min_samples_leaf  ：限定了叶子节点包含的最小样本数，这个属性对于防止上文讲到的数据碎片问题很有作用。 

模块中一些重要的属性方法：

n_classes _ ：决策树中的类数量。

classes_ ：返回决策树中的所有种类标签。

feature_importances_ ：feature的重要性，值越大那么越重要。
n_estimators: The former is the number of trees in the forest. The larger the better.has upperbound
max_features: is the size of the random subsets of features to consider when splitting a node
max_depth:  The size of each tree
bootstrap: 默认true,即有放回采样
oob_score：oob（out of band，带外）数据，即：在某次决策树训练中没有被bootstrap选中的数据。多单个模型的参数训练，我们知道可以用cross validation（cv）来进行，但是特别消耗时间，而且对于随机森林这种情况也没有大的必要，所以就用这个数据对决策树模型进行验证，算是一个简单的交叉验证。性能消耗小，但是效果不错。 
n_jobs=1： 并行job个数。这个在ensemble算法中非常重要，尤其是bagging（而非boosting，因为boosting的每次迭代之间有影响，所以很难进行并行化），因为可以并行从而提高性能。1=不并行；n：n个并行；-1：CPU有多少core，就启动多少job。  
warm_start=False：热启动，在增加新的树进去时，决定是否使用上次调用该类的结果
class_weight=None：各个label的权重。



@author: Shenjunling
"""
#%%
import numpy as np
import sklearn.datasets

#userPath = r'G:\OneDrive\tianchi\fresh_comp_offline\tianchi_fresh_comp_train_user.csv'
#A = pd.read_csv(userPath)

#%% RF 
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

boston = sklearn.datasets.load_boston()
#print boston.DESCR
#RF regression
#training
data = boston.data
target = boston.target
rf = RandomForestRegressor()#sklearn中封装了Ensemble methods.AdaBoost和bagging,RF等方法。他们都是基于同一种分类器多个不同实例的计算方法.
rf.fit(data[:300],boston.target[:300])
##预测结果差异很大
#instances = boston.data[[300, 309]]  
#print "Instance 0 prediction:", rf.predict(instances[0])  
#print "Instance 1 prediction:", rf.predict(instances[1])
#prediction, bias, contributions = ti.predict(rf, instances)#controbution每维特征的权重

#%% 用treeinterpreter 分析
#predicting
#比较两个数据集
ds1 = boston.data[300:400]  
ds2 = boston.data[400:] 
prediction1, bias1, contributions1 = ti.predict(rf, ds1) #controbution每维特征的权重
prediction2, bias2, contributions2 = ti.predict(rf, ds2)
totalc1 = np.mean(contributions1, axis=0)   
totalc2 = np.mean(contributions2, axis=0) 
print np.mean(rf.predict(ds1))  
print np.mean(rf.predict(ds2))  
print np.sum(totalc1 - totalc2)  #正好就是predicition的差异值，说明prediction等于baias加上每一维的贡献值
print np.mean(prediction1) - np.mean(prediction2)  
#把每一维特征贡献的差异之和显示出来，正好就是平均预测值的差异
for c, feature in sorted(zip(totalc1 - totalc2,   
                             boston.feature_names), reverse=True):  
    print feature, round(c, 2)  
    
    
#%% RF classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
import sklearn.datasets 


iris = sklearn.datasets.load_iris()
rfc = RandomForestClassifier(max_depth=4)
idx = range(len(iris.target))
np.random.shuffle(idx)

#%% training
#riris.data[idx[:100]]#error
rfc.fit(iris.data[idx[:100]],iris.target[idx[:100]])
ssp = StratifiedShuffleSplit(iris.target,n_iter=3,test_size=0.9,random_state=1)
for trainlab,testlab in ssp:
    print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
scores = cross_val_score(rfc,iris.data,iris.target,cv = ssp)
print("score",scores)


#%% testing
'''
   ti为每个属性计算贡献值，bias是每条样本的偏置，prediction = contribution+bias
   prediction, 
   bias, 
   contributions
'''
instance = iris.data[idx][100:]  
print rfc.predict(instance) 
prediction, bias, contributions = ti.predict(rfc, instance)#与回归不同，有多少类就有多少组bias,contribution,  
print "Prediction", prediction  #属于每一类的概率，归一化
print "Bias (trainset prior)", bias  
print "Feature contributions:"  #contribution某一列代表某一类下，各个特征的权重
for c, feature in zip(contributions[0],   
                             iris.feature_names):  
    print feature, c  
    

    