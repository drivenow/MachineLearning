# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:37:00 2016

@author: Administrator
"""
from gensim.models import Word2Vec,word2vec
from jieba import posseg 
import jieba.analyse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report,accuracy_score
from sklearn.externals import joblib
import jieba

"""
,"x":0:标点  治理
"""
def lineCut(line):
    line1=""
    filters={"nr":0,"ns":0,"nt":0,"m":0,"mq":0,"q":0,"qv":0,"qt":0,"x":0}
    tmp=posseg.cut(line)
    for w in tmp:
        ab=w.flag.encode("utf-8")      
        if (filters.has_key(ab)): 
#            print ab
            continue
        line1=line1+" "+w.word.encode("utf-8")
#        print w.word,ab
    return line1.strip()


def labClfyCode(typ,fun,clfy):
    if typ=='emot':
        if fun=='encode':
            if clfy==r'好':
                return [1,0,0]
            elif clfy==r'中':
                return [0,1,0]
            elif clfy==r'差':
                return [0,0,1]
        elif fun=='decode':
            if clfy==11:
                return r'好'
            elif clfy==10:
                return r'中'
            elif clfy==12:
                return r'差'
    elif typ=='cont':
        if fun=='encode':
            if clfy==r'动向公告':
                return [1,0,0,0,0,0]
            elif clfy==r'发展推广':
                return [0,1,0,0,0,0]
            elif clfy==r'旅游乱象':
                return [0,0,1,0,0,0]
            elif clfy==r'市场监管':
                return [0,0,0,1,0,0]
            elif clfy==r'游记攻略':
                return [0,0,0,0,1,0]
            elif clfy==r'其他':
                return [0,0,0,0,0,1]
        elif fun=='decode':
            if clfy==4:
                return r'动向公告'
            elif clfy==2:
                return r'发展推广'
            elif clfy==5:
                return r'旅游乱象'
            elif clfy==1:
                return r'市场监管'
            elif clfy==3:
                return r'游记攻略'
            elif clfy==6:
                return r'其他'

def getStopDict(stopWdsPath):
    stopWds_f=open(stopWdsPath)
    stopDict ={}
    for line in stopWds_f.readlines():
        stopDict[''.join(re.split("\s",line))]=1
    return stopDict

def tourCut(fin,fout,stopPath,splitSys="^"):
    regex=re.compile("^")
    stopWds=getStopDict(stopPath)
    f1=open(fin,'r+')
    f2=open(fout,'w+')
    lidx=0
    line=f1.readline()
    while line!="":
        if line.strip()=="":
            line=f1.readline()
            continue
        if (lidx%1)==0:
            print "**************"+str(lidx)+"**************"
        linespt=line.split("^")
        lineToWrite=linespt[0]+'^'+linespt[1]+"^"
        lineToWrite=lineToWrite+lineCut(linespt[-1])
        f2.writelines(lineToWrite+"\n")
        lidx=lidx+1
        line=f1.readline()    
    f1.close()
    f2.close()


#读取数据
"""
datain,数据路径
labed=False,是否是带标签的文本
delimeter="^"，标签的分隔符
"""
def readCorpus(datain,labed=False,delimeter="^"):
    f1=open(datain,'r+')
    corpus=[]
    lidx=0
    line=f1.readline().split("^")[-1]
    while line!="":
        if lidx%1==0:
            print "**************"+str(lidx)+"**************"
                #是否有空行    
        if line.strip()=="":
            line=f1.readline().split("^")[-1]
            continue
        subCorpus=[]
        fields=line.split(" ")
        for w in fields:
            if w.strip()!="":
                subCorpus.append(w.encode("utf-8").strip())
        corpus.append(subCorpus)
        lidx=lidx+1
        line=f1.readline()    
    f1.close()
    return np.array(corpus)


#将语料转化成word2vec模型的词向量    
def calcuCorpusVec(model,corpus,modelsize):    
    corpus_vec=[]
    for lidx,art in enumerate(corpus):
        art_vec=np.zeros(modelsize)
        cnt=0
        for word in art:
            word=word.decode("utf-8")
            if model.vocab.has_key(word):
                cnt+=1
#                print word
                art_vec+=np.array(model[word])

        if cnt>0:
            corpus_vec.append(art_vec/cnt)
        else:
            corpus_vec.append(art_vec)
        print cnt
        if lidx%1000==0:
            print "**************"+str(lidx)+"**************"
    return np.array(corpus_vec)

    

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



def rfPredict(corpus_vec,corpus_lab):
     ssp = StratifiedShuffleSplit(corpus_lab,n_iter=1,test_size=0.1)#数据集划分为验证机和测试集
     for trainlab,testlab in ssp:
         print("train:\n%s\ntest:\n%s" % (trainlab,testlab)) 
     rf=RandomForestClassifier(n_estimators=200)
     rf.fit(corpus_vec[trainlab],corpus_lab[trainlab])

     predictLab,predictProb=predLabProb(rf,corpus_vec,testlab)#在测试集上预测标签和类概率
     report=classification_report(corpus_lab[testlab],predictLab)
     scores = cross_val_score(rf,corpus_vec,corpus_lab,cv = ssp)
     acu=accuracy_score(corpus_lab[testlab],predictLab)
     return rf,report,scores,acu,trainlab,testlab


def predictNewsCtg(corpus_vec):
    rf1=joblib.load("D:/data/news/model/rf_news_ctg100.model")  
    predictLab1,predictProb1=predLabProb(rf1,corpus_vec,range(len(corpus_vec)))#在测试集上预测标签和类概率
#    report1=classification_report(corpus_lab1,predictLab1)
#    acu1=accuracy_score(corpus_lab1,predictLab1)
#    return report1,acu1
    
def predictNewsEmot():
    rf2=joblib.load("D:/data/news/model/rf_news_emot100.model")
    predictLab2,predictProb2=predLabProb(rf2,corpus_vec,range(len(corpus_vec)))

if __name__ == '__main__':
    """
    1.用Sougou构建词库
    2.用新闻样本分类word2vec模型
    3.用随机森林分类 
    """
    #%%
#    if len(sys.argv)!=5:
#        print "usage:python w2v_news_ctg.py titlein title_cut w2v_modelsize outfile"
#	sys.exit()
#    fin=sys.argv[1]
#    fout=sys.argv[2]
#    modelsize=int(sys.argv[3])
#    outfile=sys.argv[4]
#    w2vPath="model/word2vec/newsTitle.all.100"
#    rfPath="model/word2vec/rf_news_emot100.model"
#    stopPath="infile/stop.txt"
#    usrdictPath="infile/userdict.txt"
    

    fin="D:/data/news/server/news_forecast_c.txt"
    fout="D:/data/news/server/news_labeled_cut.txt"
    outfile="D:/data/news/tmp.txt"
    modelsize=100
    w2vPath="D:/data/news/model/newsTitle.all.100"
    rfPath="D:/data/news/model/rf_news_emot100.model"
    stopPath="D:/data/news/stop.txt"
    usrdictPath="D:/OneDrive/codes/python/Word2Vec/keyWords/userdict.txt"
    
    model= word2vec.Word2Vec.load(w2vPath)
    jieba.load_userdict(usrdictPath)
    tourCut(fin,fout,stopPath,"\t")
    corpus=readCorpus(fout,True)#[文本格式]：分词后的文本，一行一个文本

    #%% 存储加载模型
    print "****************calculate weibo vector**********************"
    corpus_vec=calcuCorpusVec(model,corpus,modelsize)#计算词向量

    #%% Random forest
    rf2=joblib.load(rfPath)
    predictLab2,predictProb2=predLabProb(rf2,corpus_vec,range(len(corpus_vec)))


    filewrite=file(outfile,"w")
    fileread = open(fin,"r")
    for fidx,line in enumerate(fileread): 
        line=line.strip('\n')
        if line == None:
            continue
        linespt=line.split("^")
        lineToWrite=linespt[0]+'^'+linespt[1]+"^"+linespt[2]+"^"+labClfyCode('emot','decode',int(predictLab2[fidx]))
        filewrite.writelines(lineToWrite+"\n")
    filewrite.close()


    


