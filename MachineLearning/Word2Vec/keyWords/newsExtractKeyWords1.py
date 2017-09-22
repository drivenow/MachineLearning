# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:31:11 2016

@author: Administrator
"""
from gensim.models import TfidfModel
from gensim import corpora
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("D:/OneDrive/codes/python/Word2Vec/util")
from WordVecUtil import *
from TextPre import *

#%%将新闻及对应的类别读到列表中，以便下一步提取每一类中的关键词
"""
文件内容为：分词后的文本\t文本标签;或者
            分词后的文本,没有标签（标签默认为其他）
return:ctgsList{标签名：[wordlists(一行文本一个wordlist)]}

fpath：关键词列表文件，每一行都是同义的关键词,以delimeter分割（默认文本在第0列）
delimeter:一行中关键词的分隔符
tagPlace:标签位于字段的哪个位置，<1表示文本没有标签，置为默认标签“其他”
"""
def getCtgsList(dataFile,delimeter="\t",labPlace=1):
    outf=open(dataFile,"r+")
    lines=outf.readlines()
    ctgsList={}#key:文本类别，values:文本词列表
    if labPlace>0:
        for line in lines:
            fields=line.strip().split(delimeter)
            if fields[0].strip()=="":continue
            
            ctg=fields[labPlace]
            if ctgsList.has_key(ctg)==False and ctg.strip()!="":
                ctgsList[ctg]=[]
        #每类新闻分类存放到ctgs       
        for line in lines:
            fields=line.strip().split(delimeter)
            if fields[0].strip()=="":continue
                
            ctgsList[fields[labPlace]].append(fields[0].split())
    else:
        ctgsList["未知"]=[]
        for line in lines:
            fields=line.strip().split(delimeter)
            if fields[0].strip()=="":continue
                
            ctgsList["未知"].append(fields[0].split())
    outf.close()
    return ctgsList
        
    #%%  每类新闻构建一个词库，训练一个tfidf模型，求每个类中最高词频ctgTopWord
def dictTop(freqList,dictionary,topn=20):
    n=0
    wordList=[]
    for item in freqList:
        if n>topn or n==topn:
            break    
        word=dictionary[item[0]]
        wordList.append(word+"^"+str(item[1]))#word^词频
        n=n+1
    return wordList
"""
每类新闻构建一个词库，训练一个tfidf模型，求每个类中最高词频ctgTopWord
ctgsList:{类名：[wordlists(一行文本一个wordlist)]}
"""
def getCtgTopWord(ctgsList):
    ctgTopWord={}
    for ctg in ctgsList.keys():
        dictionary = corpora.Dictionary(ctgsList[ctg])#用于生成字典
        print ctg+" dictionary items："+str(len(dictionary.items()))
        corpus=[dictionary.doc2bow(text) for text in ctgsList[ctg]]
        model=TfidfModel(corpus)
        freq=model.dfs
        ctgTopWord[ctg]=dictTop(sorted(freq.iteritems(),key=lambda a:a[1],reverse=True),dictionary,500)
    return ctgTopWord

if __name__=='__main__':
    """
    从有类别的训练样本中统计：从每个类中排名靠前的top关键词
    储存在ctgTopWord
    """
    #%%读取带标签的文本【文本格式】：文本类别“\t”分词后的文本
    stopPath="D:/data/news/stop.txt"
    basePath="D:/data/news/class"
#==============================================================================
    fin=basePath+u"/新闻六大类.txt"
    fout=basePath+u"/新闻六大类_cut.txt"
    textCut(fin,fout,labed=True,labnum=2,stopPath=stopPath)
    ctgsList=getCtgsList(fout,delimeter="\t",labPlace=1)#ctgsList{类名：[wordlists(一行文本一个wordlist)]}     
#==============================================================================

#==============================================================================
#     fin=basePath+u"/新闻标题1.9.txt"
#     fout=basePath+u"/新闻标题1.9_cut.txt"
#     textCut(fin,fout,labed=False,stopPath=stopPath)
#     ctgsList=getCtgsList(fout,delimeter="\t",labPlace=0)#ctgsList{类名：[wordlists(一行文本一个wordlist)]}     
#==============================================================================
    ctgTopWord=getCtgTopWord(ctgsList)#求每个类中最高词频ctgTopWord
    
    #%%将每类新闻的关键词写入文件
    resultPath=u"D:/data/news/key/新闻标题1.9.txt"
    resultFile=open(resultPath,"w+")
    for ctg in ctgTopWord:
        for word in ctgTopWord[ctg]:
            resultFile.write(ctg+"\t"+word.encode("utf-8")+"\n")
    resultFile.close()
    
    


    


    