# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:40:20 2016
利用词向量均值对推文进行分析效果不错，这是因为推文通常只有十几个单词，所以即使经过平均化处理仍能保持相关的特性。
一旦我们开始分析段落数 据时，如果忽略上下文和单词顺序的信息，那么我们将会丢掉许多重要的信息。
用于分析长文本
IBDM影评数据集：http://ai.stanford.edu/~amaas/data/sentiment/
@author: Administrator
"""

import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from sklearn.cross_validation import train_test_split
import numpy as np
import os 
from DocSplit import cleanText
from TextFormatInput import labelizeReviews

#%%  文本读取
class MySentenses(object):
    def __init__(self,dirName):
        self.dirName = dirName
    def __iter__(self):
        for fname in os.listdir(self.dirName):
            for line in open(os.path.join(self.dirName,fname)):
                yield line.split()
                
                
corpusSize = 500#读取的文本条数
rootPath = r"D:/data/IMDB/aclImdb/test"
fileList = os.listdir(rootPath+r'/pos')
pos_reviews = []#正面影评
i = 0
for file in fileList:
    if os.path.isfile(rootPath+r'/pos/'+file)==True and (i<corpusSize):
        line = open(rootPath+r'/pos/'+file,"r").readline()
        pos_reviews.append(line)
        i = i+1
        
fileList = os.listdir(rootPath+r'/neg')
neg_reviews = []#负面影评
i = 0
for file in fileList:
    if os.path.isfile(rootPath+r'/neg/'+file)==True and i<corpusSize:
        line = open(rootPath+r'/neg/'+file,"r").readline()
        neg_reviews.append(line)
        i = i+1
        
#fileList = os.listdir(rootPath+r'/unsup')
#unsup_reviews = []#未标记样本
#i = 0
#for file in fileList:
#    if os.path.isfile(rootPath+r'/unsup/'+file)==True and i<corpusSize:
#        line = open(rootPath+r'/unsup/'+file,"r").readline()
#        unsup_reviews.append(line)
#        i = i+1
del fileList
#%%  文本分词和划分数据集
#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
x_data, y_data, x_label, y_label = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

x_data = cleanText(x_data)#文本分词
y_data = cleanText(y_data)
#unsup_reviews = cleanText(unsup_reviews)

#%%  文本添加标签
x_data = labelizeReviews(x_data, 'TRAIN')#[文本格式]：TaggedDocument(words=['the',...'a', ' '.'], tags=['TRAIN_0'])
y_data = labelizeReviews(y_data, 'TEST')
#unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')


#%% 文本训练，得到x_train和y_train的词向量表示
"""
    The gensim documentation suggests training over the data multiple times and either adjusting the learning rate
    or randomizing the order of input at each pass.
"""
import random
import copy
#instantiate our DM and DBOW models
"""
    By default (dm=1), ‘distributed memory’ (PV-DM),Otherwise, distributed bag of words (PV-DBOW) is employed.
    )。DM 试图在给定上下文和段落向量的情况下预测单词的概率。在一个句子或者文档的训练过程中，段落 ID 保持不变，共享着同一个段落向量。
    DBOW 则在仅给定段落向量的情况下预测段落中一组随机单词的概率。
    增量训练方法：先对x_data训练，得到x文档向量；再对y_data增量训练，得到y文档向量
"""
size = 80
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    
#合并语料
corpus = copy.deepcopy(x_data) 
corpus.extend(y_data)
#corpus.extend(unsup_reviews)
model_dm.build_vocab(corpus) 

all_train_reviews = copy.deepcopy(x_data) 
#all_train_reviews.extend(unsup_reviews)
#all_train_reviews = np.concatenate((x_data, unsup_reviews))
#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
for epoch in range(4):
    random.shuffle(all_train_reviews,random=random.seed(epoch))
    random.shuffle(x_label,random = random.seed(epoch))
    model_dm.train(all_train_reviews)

#Get training set vectors from our models
"""
   doc2vec：根据文本标签获取文本特征向量获取：model.docvecs["TRAIN_1"]
   训练次数：
"""
def getVecs(model, corpus, size):
    docs_num = len(corpus)
    train_array = np.zeros((docs_num,size))
    for i in range(docs_num):
        train_array[i] = model.docvecs[corpus[i].tags]#corpus[i].tags文本标签
    return train_array
train_vecs_dm = getVecs(model_dm, all_train_reviews, size)

#train over test set
for epoch in range(4):
    random.shuffle(y_data,random=random.seed(epoch))
    random.shuffle(y_label,random=random.seed(epoch))
    model_dm.train(y_data)

#Construct vectors for test reviews
test_vecs_dm = getVecs(model_dm, y_data, size)


#%%  换一种训练方法dm=0，两个训练方法的词向量，组成联合特征
#model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
#model_dbow.build_vocab(corpus)
#del corpus
#
#
#for epoch in range(1):
#    random.shuffle(all_train_reviews)
#    model_dbow.train(all_train_reviews)
#del all_train_reviews
#train_vecs_dbow = getVecs(model_dbow, all_train_reviews, size)
#train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
##train over test set
#for epoch in range(1):
#     random.shuffle(y_data)
#     model_dm.train(y_data)
#
##Construct vectors for test reviews
#test_vecs_dbow = getVecs(model_dbow, y_data, size)
#test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
#test_vecs_dm =test_vecs
#train_vecs_dm = train_vec

#%%  线性分类器
from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier(loss='log', penalty='l1')
classifier.fit(train_vecs_dm, x_label)

print 'Test Accuracy: %.2f'%classifier.score(test_vecs_dm, y_label)

#%%
##%% 随机森林分类器
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report
#classifier = RandomForestClassifier(n_estimators=100,n_jobs=3,oob_score=True,bootstrap=True)
#classifier.fit(train_vecs_dm, x_label)
#print classification_report(classifier.predict(train_vecs_dm), x_label)
#print 'Test Accuracy: %.2f'%classifier.score(test_vecs_dm, y_label)


#%%  画ROC曲线
#Create ROC curve
from sklearn.metrics import roc_curve, auc
#matplotlib inline
import matplotlib.pyplot as plt

pred_probas = classifier.predict_proba(test_vecs_dm)[:,1]

fpr,tpr,_ = roc_curve(y_label, pred_probas)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()