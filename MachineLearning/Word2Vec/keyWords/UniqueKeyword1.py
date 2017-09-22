# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:18:56 2017

@author: Administrator
"""

import re 
import os
regex=re.compile("\n")
    
"""
#数据去重
    将wordList中首次出现的词，加入到keyWordDict
"""    
def keyWordUnique(keyWordDict,wordsList):
    for word in wordsList:
        if keyWordDict.has_key(word):
            continue
        else:
            keyWordDict[word]=0
    return keyWordDict
    
"""
从文件中获取关键词
"""
def getKeywordDict(fpath):
    fin=open(fpath,"r")
    keywordDict={}
    for line in fin:
        print line
        wordsList=line.split("\t")
        keywordDict=keyWordUnique(keywordDict,wordsList)
    fin.close()
    return keywordDict
    
    
inBasePath=u"D:/data/news/key/merge"
files=os.listdir(inBasePath)

for fname in files:
    keyWordDict=getKeywordDict(inBasePath+"/"+fname)
    fout=open(inBasePath+"/unique"+fname,"w") 
    for kidx,keyWord in enumerate(keyWordDict.keys()):
        #设计过滤规则
        if re.search("^\w",keyWord)!=None:#过滤数字字母开头
            continue

        if re.search("\n",keyWord)!=None:
            print "keyWord\n: "+keyWord
            fout.write(keyWord)
        else:
            fout.write(keyWord+"\t")
    fout.write("\n")
    fout.close()
            