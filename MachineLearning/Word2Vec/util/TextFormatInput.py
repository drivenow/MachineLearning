# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:09:21 2016

@author: Administrator
"""
import gensim
from gensim.corpora import textcorpus

LabeledSentence = gensim.models.doc2vec.LabeledSentence
#Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
#We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
#a dummy index of the review.
"""
   LabeledSentence:输入对象是（短文本，标签），输出对象是有两个字段：words和tags
   LabeledSentence:'/Volumes/Macintosh HD/Users/RayChou/Downloads/情感分析训练语料/neg_train.txt':'TRAIN_NEG',
"""
"""
   对doc2vec的文档加标签
   reviews:评论内容的列表
   label_type:标签的前缀字符串，后缀是递增数字
"""
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

"""
corpus中的默认分词工具，都是讲文档转化成bow向量（稀疏词向量）
#默认根据空格进行分词,适合英文文本资料输入
"""    
text=textcorpus.TextCorpus("D:/data/SougouNews/resultbig.txt")
dct = text.dictionary#字典迭代对象
dct[1]#单词


#%% 
"""
迭代器，一行一行返回文本
"""
def textIter(filename):
    f=open(filename,"r+")
    for line in f:
        yield line
        
ti=textIter("D:/data/SougouNews/resultbig.txt")
q=ti.next()    
