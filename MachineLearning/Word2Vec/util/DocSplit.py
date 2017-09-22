# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:05:08 2016

@author: Administrator
"""
import jieba
import jieba.posseg as pseg
import re

stopWdsPath= r"d:\data\stopWords.txt"

#Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

"""
精确模式:默认，
全模式：cut_all=true，会对所有分词的组合一遍，看是不是新词
搜索引擎模式：jieba.cut_for_search(),会对词进行一些近义词替换，
jieba.load_userdict("userdict.txt"),（word,frequency,词性）同理可设置停用词表
jieba.add_word('凱特琳')
jieba.analyse.extract_tags(content, topK=topK),基于TFIDF算法，还有基于TextRank的算法

"""
#seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
def jiebaText(corpus):
    stopWds_f=open(stopWdsPath)
    stopDict ={}
    for line in stopWds_f.readlines():
        stopDict[''.join(re.split("\s",line))]=1
    nwordAll=[]
    for corpu in corpus:
        #取出特定词性的词,去除停用词
        words = pseg.cut(corpu)#(单词，词性)
        nword = []
        for w in words:  
            ab=w.flag.encode("utf-8")
            if((ab == 'n'or ab == 'v' or ab == 'a') and len(w.word.encode("utf-8"))>1
            and not stopDict.has_key(w.word.encode("utf-8"))):    
                nword.append(w.word.encode("utf-8"))
        nwordAll.append(nword)
    return nwordAll
        