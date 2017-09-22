# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:35:40 2016

@author: Administrator
"""
from jieba import posseg
import re
import numpy as np
import os
import jieba.analyse
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
#sys.path.append("D:/OneDrive/codes/python/Word2Vec/util")
#from WordVecUtil import *
#from TextPre import *


"""
对一行文本分词
返回：以空格分隔的文本
    ,"x":0:标点,p:介词，q:量词
    词性过滤：filters={"nr":0,"ns":0,"nt":0,"m":0,"mq":0,"q":0,"qv":0,"qt":0}
"""
def lineCut(line,filters={},stopPath=None):
    if stopPath!=None:
        stopWds=getStopDict(stopPath)
    else:
        stopWds={}
    line1=""
    tmp=posseg.cut(line)
    for w in tmp:
        if(stopWds.has_key(w.word.encode("utf-8"))==True):
            print "**************stopWds: "+w.word+"**************"
            continue
        ab=w.flag.encode("utf-8") 
        if (filters.has_key(ab)): 
            print "**************filter: "+w.word+ab+"**************"
            continue#过滤词性
        line1=line1+" "+w.word.encode("utf-8")
    return line1.strip()

"""
textCut(fin,fout,labed=False,labnum=1,delimeter="\t",filters=[],stopPath=None):  
return: None
"""
def textCut(fin,fout,labed=False,labnum=1,delimeter="\t",filters=[],stopPath=None):  
    f1=open(fin,'r+')
    f2=open(fout,'w+')
    lidx=0
    line=f1.readline()
    while line!="":
        if line.strip()=="":
            line=f1.readline()
            continue
        if (lidx%10)==0:
            print "**************textCut:"+str(lidx)+"**************"
        if labed==True:
            source=line.split(delimeter)
            line=source[0].strip()
            lab1=source[1].strip()
            if labnum==2:
                lab2=source[2].strip()
            else:
                lab2=''
            lab=delimeter+lab1+delimeter+lab2#文本原标签
        line_c=lineCut(line,stopPath=stopPath)#分词后文本 
        if labed==True:
            f2.writelines(line_c+lab+"\n")
        else:
            f2.writelines(line_c+"\n")
        lidx=lidx+1
        line=f1.readline()    
    f1.close()
    f2.close()


#读取数据
"""
datain,数据路径
cutted=True,是否是已经分词的文本
labed=False,是否是带标签的文本
labnum=1,labed==true时有效，标签的个数（1或2个）
delimeter="^"，labed==true时有效，标签的分隔符
"""
def readCorpus(datain,cutted=True,labed=True,labnum=1,delimeter="\t"):
    f1=open(datain,'r+')
    corpus=[]
    corpus_lab1=[]
    corpus_lab2=[]
    lidx=0
    line=f1.readline()
    while line!="":
        if lidx%10==0:
            print "**************readCorpus:"+str(lidx)+"**************"  
        if line.strip()=="":
            line=f1.readline()#去除空行
            continue
        if labed==True:
            print line
            source=line.split(delimeter)
            line=source[0]#文本
            if line.strip()=="":
                line=f1.readline()
                continue
            assert len(source)==2,"fileds' lenght is not 2 at "+str(lidx)#保证有两个字段
            line_lab1=source[1]#标签1
            corpus_lab1.append(line_lab1.strip())
            if labnum==2:
                assert len(source)==3,"fileds' lenght is not 3 at "+str(lidx)#保证有三个字段
                line_lab2=source[2]#标签2     
                corpus_lab2.append(line_lab2.strip())
        #是否是分此后的文本
        if cutted==True:
            subCorpus=[]
            fields=line.split(" ")
            for w in fields:
                if w.strip()!="":
                    subCorpus.append(w.encode("utf-8").strip())
            corpus.append(subCorpus)#文本按空格fen
            lidx=lidx+1
            line=f1.readline()    
        else:
            lidx=lidx+1
            corpus.append(line.strip())#文本直接输出
            line=f1.readline()
    f1.close()
    if labed==True:
        if labnum==1:
            return np.array(corpus),np.array(corpus_lab1)#只有一个标签
        else:
            print corpus_lab1
            return np.array(corpus),np.array(corpus_lab1),np.array(corpus_lab2)
    else:
        return corpus


#读取sougou新闻数据集
def readSougou(indir):
    files=os.listdir(indir)
    corpus=[]
    lidx=0
    for f in files:
        if lidx%10000==0:
            print "**************Sougou: "+str(lidx)+"**************"
        f1=open(f,'r+')
        subCorpus=[]
        line=f1.readline()
        while line!="":
            subCorpus.append(line.strip())
            line=f1.readline()
        corpus.append(subCorpus)
        lidx=lidx+1
        f1.close()
    return corpus

"""
获得停用词词典
返回：停用词列表,str(非unicode编码)
    停用词文件格式：一行一个词
"""   
def getStopDict(stopWdsPath):
    stopWds_f=open(stopWdsPath)
    stopDict ={}
    for line in stopWds_f.readlines():
        stopDict[''.join(re.split("\s",line))]=1
    return stopDict

"""
将list写入文件
"""
def writeList(filename,lista):
    f1=open(filename,"w+")
    for ele in lista:
        f1.writelines(ele)
        f1.writelines("\n")
    f1.close()
    
  
"""
"D:/data/comment/pred_trust.txt"
#根据阈值将置信度低的文本，作为手动分类的文本写入doubt文件；将置信度高的写入trust文件 
返回：文件【语料\t预测的标签\t真实的标签\n】
corpus_source,predictLab,predictProb,trueLab共享同一索引，即同一位置是同一文本的值
thea:置信度阈值，predictProb中可能性低于该值的，写入doubt文件
""" 
def doubt2file(trustPath,doubtPath,corpus_source,predictLab,predictProb,trueLab,thea):
    trust=[]
    doubt=[]
    for pidx,prob in enumerate(predictProb):
#        print max(prob)
        if max(prob)>thea:   
            trust.append((pidx,max(prob)))
        else:
            doubt.append((pidx,max(prob)))
    
    f1=open(trustPath,"a")
    f2=open(doubtPath,"a")
    for tidx,tprob in enumerate(trust):
        f1.write(str(tidx)+"\t"+corpus_source[tidx].strip().encode("utf-8")+"\t"+str(predictLab[tidx])+"\t"+str(tprob)+"\t"+str(trueLab[tidx])+"\t"+str(times)+"\n")
    for didx,dprob in doubt:
        f2.write(str(didx)+"\t"+corpus_source[didx].strip().encode("utf-8")+"\t"+str(predictLab[didx])+"\t"+str(dprob)+"\t"+str(trueLab[didx])+"\t8"+"\n")
    f1.close()
    f2.close()
    
def labClfyCode(typ,fun,clfy):
    print clfy
    if typ=='emot':
        if fun=='encode':
            if clfy==u'好':
                return 1
            elif clfy==u'中':
                return 0
            elif clfy==r'差':
                return 2
#        elif fun=='decode':
#            if clfy==1:
#                return r'好'
#            elif clfy==0:
#                return r'中'
#            elif clfy==2:
#                return r'差'
        elif fun=='decode':
            if clfy==0:
                return r'好'
            elif clfy==1:
                return r'中'
            elif clfy==2:
                return r'差'
    elif typ=='cont':
        if fun=='encode':
            if clfy==r'动向公告':
                return 4
            elif clfy==r'发展推广':
                return 2
            elif clfy==r'旅游乱象':
                return 5
            elif clfy==r'市场监管':
                return 1
            elif clfy==r'游记攻略':
                return 3
            elif clfy==r'其他':
                return 6
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
                
def oneHotClfyCode(typ,fun,clfy):
    if typ=='emot':
        if fun=='encode':
            if clfy==r'好':
                return [1,0,0]
            elif clfy==r'中':
                return [0,1,0]
            elif clfy==r'差':
                return [0,0,1]
        elif fun=='decode':
            if sum(abs(clfy-[1,0,0]))==0:
                return r'好'
            elif sum(abs(clfy-[0,1,0]))==0:
                return r'中'
            elif sum(abs(clfy-[0,0,1]))==0:
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
            if sum(abs(clfy-[1,0,0,0,0,0]))==0:
                return r'动向公告'
            elif sum(abs(clfy-[0,1,0,0,0,0]))==0:
                return r'发展推广'
            elif sum(abs(clfy-[0,0,1,0,0,0]))==0:
                return r'旅游乱象'
            elif sum(abs(clfy-[0,0,0,1,0,0]))==0:
                return r'市场监管'
            elif sum(abs(clfy-[0,0,0,0,1,0]))==0:
                return r'游记攻略'
            elif sum(abs(clfy-[0,0,0,0,0,1]))==0:
                return r'其他'

#times=times+1
#trustPath="D:/data/news/error/cont_high.txt"
#doubtPath="D:/data/news/error/cont_low.txt" 
#corpus_source=xTestConText
#predictLab=modelCont.predict_classes(xTestCont)
#predictLab=[labClfyCode("cont","decode",lab+1) for lab in predictLab]
#predictProb=contClfyPredict
#trueLab=np.array([oneHotClfyCode("cont","decode",lab) for lab in yTestCont])
#doubt2file(trustPath,doubtPath,corpus_source,predictLab,predictProb,trueLab,thea=0.6)
    
    
#预测正确与不正确的编号
def predictYN(predictLab,corpus_lab):
    predictY=[]
    predictN=[]
    for tidx,tlab in enumerate(corpus_lab):
        if (predictLab[tidx]!=tlab):
           predictY.append(tidx)
        if (predictLab[tidx]==tlab):
            predictN.append(tidx)
    return  predictY,predictN

"""
加入idf词信息，提取文章的关键词
"""
def keyWordExt(content,top,withWeightEn,rate):
    tags = jieba.analyse.extract_tags(content, topK=top, withWeight=withWeightEn)
    tagCnt=0
    keyWord=[]
    baseVal=0.1
    for tag in tags:
        if tagCnt==1:
            keyWord.append(tag[0].encode("utf-8"))
            baseVal=tag[01]
        else:
            if tag[01]>=baseVal*rate :
                keyWord.append(tag[0].encode("utf-8"))
        tagCnt+=1
    return keyWord


