# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 16:23:15 2016

@author: Administrator
"""

from gensim.models import word2vec
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from keras.models import Model
from keras.layers import Dense,Input
from keras.utils import np_utils
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("D:/OneDrive/codes/python/Word2Vec/util")
from WordVecUtil import *
from TextPre import *

#在训练集上预测标签和类概率
"""
    rf：模型
    corpus_vec：文本的word2vec向量
    testlab:测试集在corpus_vec中的编号
"""
def predLabProb(rf,corpus_vec,testlab):
    predictProb=rf.predict_proba(np.array(corpus_vec)[testlab])
    predictLab=rf.predict(np.array(corpus_vec)[testlab])
    return predictLab,predictProb
    
    
def nnPredict(corpus_vec,corpus_lab):
    nn=MLPClassifier(hidden_layer_sizes=(500,50,10),activation="relu",max_iter=1000000)
    ssp = StratifiedShuffleSplit(corpus_lab,n_iter=1,test_size=0.1,random_state=1)#数据集划分为验证机和测试集
    for trainlab,testlab in ssp:
        print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
    nn.fit(corpus_vec,corpus_lab)
    predictLab,predictProb=predLabProb(nn,corpus_vec,testlab)
    report=classification_report(np.array(corpus_lab)[testlab],predictLab)
    scores = cross_val_score(nn,corpus_vec,corpus_lab,cv = ssp)
#        joblib.dump(rf,"D:/data/comment/model/RandomForest_emot.model")#存储
    return nn,report,scores
    

def datasetSplit(corpus_vec,corpus_lab):
    ssp = StratifiedShuffleSplit(corpus_lab,n_iter=1,test_size=0.30,random_state=1)
    for trainlab,testlab in ssp:
        print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
    X_train=corpus_vec([trainlab])
    X_test=corpus_vec([testlab])
    Y_train=np_utils.to_categorical(corpus_lab[trainlab])
    Y_test=np_utils.to_categorical(corpus_lab[testlab])
    return X_train,X_test,Y_train,Y_test

if __name__ == '__main__':
    """
    对word2vec分类，并对分类置信度低的样本
    神经网络准确率只有70%出头，matlab上也是如此，怎么调节参数（或者改变网络结构）？
    """
    #分词预处理
#    fin="D:/data/comment/comment1_labed.txt"
#    fout="D:/data/comment/comment1_labed_cut.txt"
#    tourCut(fin,fout)
    #%%
    datain="D:/data/comment/comment1_labed_cut.txt"
    corpus,corpus_lab=readCorpus(datain,cutted=True,labed=True,labnum=1,delimeter="^")
#    print "****************build_vocab**********************" 
    modelsize=50
    model = word2vec.Word2Vec(size=modelsize,min_count=1, window=5)  #训练skip-gram模型; 默认window=5  
    model.build_vocab(corpus)
    model.train(corpus,total_examples=len(corpus))    #模型（增量）训练
    #%% 存储加载模型
#==============================================================================
#    w2v = word2vec.Word2Vec.load(r'D:/data/comment/model/word2vec50_emot')
#==============================================================================
    print "****************calculate weibo vector**********************"
    corpus_vec=calcuCorpusVec(w2v,corpus,modelsize,udecode=False)
    np.savetxt("D:/data/corpus_vec.txt",corpus_vec,fmt="%.4f",delimiter=",")
    #%% Random forest
    nn,report,scores=nnPredict(corpus_vec,corpus_lab)       
    

