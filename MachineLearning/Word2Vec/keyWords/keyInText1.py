# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:19:04 2016
@author: Administrator
ctgDict{类别：{关键词字典}}
判断标题中包含哪类、哪个关键字
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("D:/OneDrive/codes/python/Word2Vec/util")
from WordVecUtil import *
from TextPre import *

#%% 加载关键词表
"""
(关键词文件)
从文件中读取关键词，文件名：eg:发展推广final.txt，文件内容：关键词\t关键词
return:ctg(类名)，keyDict(关键词字典)

fpath：关键词列表文件，每一行都是同义的关键词,以delimeter分割
delimeter:一行中关键词的分隔符

"""
def loadKeyFile(fpath,delimeter="\t"):   
    f=open(fpath,"r")
    lines=f.readlines()
    ctg=fpath.split("/")[-1].split(".")[0][0:-5]
    keyDict={}#储存某一类的关键词
    for line in lines:
        line=line.strip()
        if line=="":
            continue
        fields=line.split(delimeter)
        for field in fields:
            if field.strip()!="":
                keyDict[field]=0
    return ctg.encode("utf-8"),keyDict

"""
从文件中读取关键词，文件名为关键词的类，文件内容为：关键词\t关键词
return:ctgsDict{类名：{关键词：0}}

fpath：关键词列表文件，每一行都是同义的关键词,以delimeter分割（默认文本在第0列）
delimeter:一行中关键词的分隔符
tagPlace:标签位于字段的哪个位置
"""
def loadKeyFileList(fileList):
    ctgDict={}#储存所有类的关键词
    for fpath in fpathList:
        ctg,keyDict=loadKeyFile(fpath,"\t")
        ctgDict[ctg]=keyDict
    return ctgDict


#%% 根据新闻标题中存在的关键词，分类
"""
#每个类别写一份文档
ctgDict:每类中的关键词
return:
result_include:哪些文本包含分类关键词
result_declude:哪些文本不包含分类关键词
"""
def keyIntitle(inlines,ctgDict):
    result_include=[]#包含关键词的标题
    result_declude=[]#不包含关键词的标题
    for line in inlines:
        content="".join(line)
        flag=False#判断是否有关键词
        outline=""#result_include中包含的行

        for ctg in ctgDict.keys():
            wordsList=[]#该ctg类中包含的词
            topWords=ctgDict[ctg]#某类的关键词
            subflag=False#判断是否有该类的关键词
            for word in topWords:       
                if content.find(word)!=-1:
                    flag=True
                    subflag=True
                    wordsList.append(word)
            if subflag==True:
                outline=outline+ctg+"\t"+" ".join(wordsList)+"\t"

        if flag==False:
            result_declude.append(content+"\t"+"未知"+"\n")
        else:
            result_include.append(content+"\t"+outline+"\n")
    return result_include,result_declude


"""
根据{类别：关键词列表}，判断哪些文本，是否包含关键词，包含的是哪一类的关键词
"""
#==============================================================================
fpathList=[u"D:/data/news/key/UniqueMerge/旅游动态merge.txt",
           u"D:/data/news/key/UniqueMerge/旅游乱象merge.txt",
           u"D:/data/news/key/UniqueMerge/其他merge.txt",
           u"D:/data/news/key/UniqueMerge/市场监督merge.txt",
           u"D:/data/news/key/UniqueMerge/游记攻略merge.txt"]
#==============================================================================
ctgDict=loadKeyFileList(fpathList)          
datain=u"D:/data/news/class/新闻标题1.9.txt"
inlines=readCorpus(datain,cutted=False,labed=False,labnum=1,delimeter="\t")
result_include,result_declude=keyIntitle(inlines,ctgDict)

            
            
            