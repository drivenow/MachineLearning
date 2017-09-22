# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 16:23:15 2016

@author: Administrator
"""
"""

"""
from gensim.models import Word2Vec,word2vec
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import os
import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("D:/OneDrive/codes/python/Word2Vec/util")
from WordVecUtil import *
from TextPre import *

"""
#模型训练
#文本格式:[一行一个文本，每个文本中单词用空格分割]    
#maxLength:文本的个数
"""
def w2vTrain(filePath,maxLength=100000):
    modelsize=60
    model = word2vec.Word2Vec(size=modelsize,min_count=2, window=7)  #训练skip-gram模型; 默认window=5  
    sougou=word2vec.LineSentence(filePath,max_sentence_length=maxLength)
    model.build_vocab(sougou)
    model.train(sougou)    #模型（增量）训练
    model.save(r"D:/data/news/model/word2vec"+str(modelsize)+"_emot")
    return model
    
"""
文件内容为：分词后的文本\t文本标签;或者
          分词后的文本,没有标签（标签默认为其他）
            
return:ctgsList{类名：[wordlists(一行文本一个wordlist)]}

fpath：关键词列表文件，每一行都是同义的关键词,以delimeter分割（默认文本在第0列）
delimeter:一行中关键词的分隔符
tagPlace:标签位于字段的哪个位置，<1表示文本没有标签
"""
def getCtgsList(dataFile,delimeter="\t",labPlace=1):
    outf=open(dataFile,"r+")
    lines=outf.readlines()
    ctgsList={}#key:文本类别，values:文本词列表
    if labPlace>0:
        for line in lines:
            fields=line.strip().split(delimeter)
            if line.strip()=="":continue
            
            ctg=fields[labPlace].strip()
            if ctgsList.has_key(ctg)==False and ctg.strip()!="":
                ctgsList[ctg]=[]
        #每类新闻分类存放到ctgs       
        for line in lines:
            print line
            fields=line.strip().split(delimeter)
            if line.strip()=="":continue 
            ctgsList[fields[labPlace].strip()].append(fields[0].split())
    else:
        ctgsList["未知"]=[]
        for line in lines:
            fields=line.strip().split(delimeter)
            if fields[0].strip()=="":continue
            ctgsList["未知"].append(fields[0].split())
    outf.close()
    return ctgsList
        



if __name__ == '__main__':
    """
    利用训练好的word2vec文本模型，
    寻找关键词的相似词,存入*s.txt文件
    """
    jieba.load_userdict("D:/OneDrive/codes/python/Word2Vec/keyWords/userdict.txt")    
    #%%模型训练
#==============================================================================
#    filePath="D:/data/SougouNews/sougouNewsMerge.txt"
#     model=w2vTrain(filePath,maxLength=480000)
#==============================================================================
    #%% 存储加载模型
#==============================================================================
    w2v = word2vec.Word2Vec.load(r'D:/data/news/model/newsAll.model.vec80')
#==============================================================================
#%%
    #各个类别的基本关键词
    fout=u"D:/data/news/class/新闻五大类_提炼.txt"
    ctgsList=getCtgsList(fout,delimeter="\t",labPlace=1)
    """
    #寻找关键词的相似词,两层字典结构，{ctg:{keyword:similar words}}，写入文件
    """
    kDict={}#寻找关键词的相似词,两层字典结构，{ctg:{keyword:similar words}}
    for ctg in ctgsList.keys():
        kLines=ctgsList[ctg]
        kDict[ctg]={}
        outf=open(u"D:/data/news/class/"+ctg+".txt","w")
        for wordsList in kLines:
            similar_str_dict=mostSimilarWordsList(w2v,wordsList,topn=20)
            for keyWord in similar_str_dict.keys():
                outf.write(keyWord+"\t"+similar_str_dict[keyWord]+"\n")
                kDict[ctg][keyWord]=similar_str_dict[keyWord]
        outf.close()
  #%%     
        """
    #数据去重
        将wordList中首次出现的词，加入到keyWordDict
        """
    import re 
   
    def keyWordUnique(keyWordDict,wordsList):
        for word in wordsList:
            if keyWordDict.has_key(word):
                continue
            else:
                keyWordDict[word]=0
        return keyWordDict
        
    inBasePath=u"D:/data/news/key/merge"
    outBasePath=u"D:/data/news/key/UniqueMerge"
    files=[u"D:/data/news/key/merge/旅游动态merge.txt",
           u"D:/data/news/key/merge/旅游乱象merge.txt",
           u"D:/data/news/key/merge/其他merge.txt",
           u"D:/data/news/key/merge/市场监督merge.txt",
           u"D:/data/news/key/merge/游记攻略merge.txt"]
    
    #对每个文件生成一个去重后的文件
    for fname in files:
        keyWordDict={}
        fin=open(fname,"r")
        fout=open(outBasePath+"/Unique"+fname.split(fname)[-1],"w") 
        for line in fin:
            print line
            wordsList=line.split("\t")
            keyWordDict=keyWordUnique(keyWordDict,wordsList)#对字典元素去重
        for kidx,keyWord in enumerate(keyWordDict.keys()):
            #设计过滤规则
            if re.search("^\w",keyWord)!=None:#过滤数字开头
                continue
            #写文件
            if re.search("\n",keyWord)!=None:
                print "keyWord in keyWordDict: "+keyWord
                fout.write(keyWord)
            else:
                fout.write(keyWord+"\t")
        fout.write("\n")
        fin.close()
        fout.close()
            

        
                
        

