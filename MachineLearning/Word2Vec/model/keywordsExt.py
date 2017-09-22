#coding:utf-8
import jieba
import jieba.analyse
from optparse import OptionParser
import sys
import time
import os

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

#infile=sys.argv[1]
#outfile=sys.argv[2]
#idfFile=sys.argv[3]
infile=sys.argv[1]
outfile=sys.argv[2]
idfFile=sys.argv[3]

#set idf path
filewrite=open(outfile,"w")
fileread = open(infile,"r")

if  os.path.exists(idfFile):
    jieba.analyse.set_idf_path(idfFile)
    
for line in fileread:
    line=line.strip('\n')
    if line == None:
        continue
    linespt=line.split("^")
    keyWords=keyWordExt(linespt[-1],10,True,0.7)
    outline="^".join(keyWords)
    filewrite.writelines(outline+"\n")
    
fileread.close()
filewrite.close()

