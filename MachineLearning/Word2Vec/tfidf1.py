#encoding:utf-8
'''
Created on 2015年10月25日

@author: Administrator
'''
""

from gensim import corpora, models
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("D:/OneDrive/codes/python/Word2Vec/util")
from WordVecUtil import *
from TextPre import *


def saveIdf(tfidfModel,idfOutFile):
    idff=open(idfOutFile,"w")
    idfdict=sorted(tfidf.idfs.iteritems(),key=lambda x:x[1],reverse=True)
    for lidx,idfTuple in enumerate(idfdict):
        print lidx,idfTuple
        out=dictionary[idfTuple[0]].strip()+" "+str(idfTuple[1])+"\n"
        idff.write(out)
    idff.close()
    
def saveTf(tfidfModel,tfOutFile):
    tff=open(tfOutFile,"w")
    tfdict=sorted(tfidf.dfs.iteritems(),key=lambda x:x[1],reverse=True)
    for listele in tfdict:
        out=dictionary[listele[0]].strip()+" "+str(listele[1])
        tff.writelines(out+"\n")
    tff.close()


#%% 读取分词后的数据集
datain=u"D:/data/news/新闻标题_分词.txt"
dataset=readCorpus(datain=datain,cutted=True,labed=False)

#%% tfidf加权  
dictionary = corpora.Dictionary(dataset)#用于生成字典
corpus = [dictionary.doc2bow(text) for text in dataset]#doc2bow将文本转换成稀疏向量
tfidf = models.TfidfModel(corpus)  
#==============================================================================
# print tfidf.dfs #(wordno,count) 
# print tfidf.idfs #(wordno，idf) 
# corpus_tfidf = tfidf[corpus]
#==============================================================================

#储存idf文件
idfOutFile="D:/data/news/tfidf/titleIdf.txt"
saveIdf(tfidf,idfOutFile)
#储存tf文件
tfOutFile="D:/data/news/tfidf/titleTf.txt"
saveTf(tfidf,tfOutFile)




