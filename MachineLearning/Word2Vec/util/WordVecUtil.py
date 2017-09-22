# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:48:44 2016

@author: Administrator
"""
import numpy as np
import sys
sys.setdefaultencoding('utf-8')
from six import string_types
"""
将语料转化成word2vec模型的词向量 
返回：列表[每一篇文章对应一个词向量]
    model:已加载的为w2v模型
    corpus:列表，语料中的一个个单词
    modelsize:w2v词向量维数
    udecode:contrain vec==0时，考虑将edecode设置为true或false.w2v的字典中的词一般是非utf8编码的
"""   
def calcuCorpusVec(model,corpus,modelsize,udecode=True):    
    corpus_vec=[]
    for lidx,art in enumerate(corpus):
        art_vec=np.zeros(modelsize)
        cnt=0
        for word in art:
            if udecode==True:
                word=word.decode("utf-8")
            if model.vocab.has_key(word):
                cnt+=1
#                print word
                art_vec+=np.array(model[word])

        if cnt>0:
            corpus_vec.append(art_vec/cnt)
        else:
            corpus_vec.append(art_vec)
        print "contrain vec:"+str(cnt)
        if lidx%1000==0:
            print "**************CorpusVec： "+str(lidx)+"**************"
    return np.array(corpus_vec)
    
    

"""
寻找词向量最相似的那些文本
返回：最相似的topidx(语料编号),topdis(语料距离)
    tvec:待匹配向量
    tid：待匹配向量在检索库中编号
    topn:排名前几的
    title_vec：待检索语料库，对应的向量
"""
def distance(vector1,vector2):  
    d=0;  
    for a,b in zip(vector1,vector2):  
        d+=(a-b)**2;  
    return d**0.5;
def mostsim(tvec,tid,topn,title_vec):
    topidx=np.zeros(topn)#标题编号
    topdis=np.ones(topn)*100#与该标题的距离
    maxi=0
    maxdis=100
    n=0
    for vec in title_vec:
        dis=distance(vec,tvec)
        if dis<maxdis and n!=tid:
            topdis[maxi]=dis
            topidx[maxi]=n#标题的编号 
        
        maxdis=topdis[0]
        maxi=0
        for subi,subdis in enumerate(topdis):
            if(subdis>maxdis):
                maxdis=subdis
                maxi=subi#在top列表中的编号     
        n=n+1
    return topidx,topdis
    

"""
将最相似的对应的语料找出，写入文件
返回：outpath文件
    outpath：输出文件
    title:语料库，（每篇文章是分词列表）
    midx:最相似的文本编号
    mdis:最相似的文本距离
    t:被匹配的向量对应的分词列表
"""
def list2str(lista,delimiter=" "):
    stra=""
    for a in lista:
        stra=stra+delimiter+a.encode("utf-8")
    return stra
    
def sim2Text(outpath,tcut,titles,midx,mdis):        
    outfile=open(outpath,"w")
    outfile.write(list2str(tcut)+"\n")
    for i in midx:
        toptitle=list2str(titles[int(i)])+"\n"
        outfile.write(toptitle)
    outfile.close()

"""
给出关键词列表，寻找w2v中最相似的topN个词
返回：最相似词列表，用空格分割
"""
def writeKDict(keyOutFile,keyDict):
    kf=open(keyOutFile,"w+")
    for key in keyDict:
        kf.write(key+"\t"+keyDict[key]+"\n")
    kf.close()
        
def mostSimilarWordsList(model,keyWordList,topn=10): 
    kDict={}
    for word in keyWordList:
        print word
        word=word.strip().decode("utf-8")
        if word=="":continue
        if model.vocab.has_key(word):
            similar_tuple=model.most_similar(positive=word,topn=topn)
            similar_list=[t[0] for t in similar_tuple]
            similar_str="\t".join(similar_list)
            print "similar_str: "+similar_str
            kDict[word.decode("utf-8")]=(similar_str)
    return kDict

