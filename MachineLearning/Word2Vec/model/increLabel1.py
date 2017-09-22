# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:04:52 2016
@author: Administrator
"""
from gensim import corpora, models, similarities
import numpy as np
import jieba.posseg as pseg
import os
import time
import sys
import re 
reload(sys)
sys.setdefaultencoding("utf-8")
import datetime
import time

#%%  jieba分词
"""
articlesPath：不带标签的纯标题文本文件
"""
def readCorpus(articlesPath,lineno):
    lines = open(articlesPath,"r").readlines()
    cut_dataset = []
    dataset= []
    start=time.clock()
    for lidx,line in enumerate(lines):
        if lidx>lineno:break
        line=line.replace("\n","")   
#        print sublines.encode("utf-8")  
        if lidx%1000==0:
            print "*************cut Corpus"+str(lidx)+"*************"
        cut_line=[]
        tmp=pseg.cut(line)
        for w in tmp:
            ab=w.flag.encode("utf-8")
            #去除某些类型的词
            if ((ab == 'n'or ab == 'v' or ab == 'a') and len(w.word.encode("utf-8"))>1): 
                cut_line.append(w.word.encode("utf-8"))
        dataset.append(line)
        cut_dataset.append(cut_line)
    end=time.clock()
    print " %f s" % (end - start)
    return dataset,cut_dataset


#%% tfidf加权 
"""
合并历史的词频文件，
cut_dataset,
tfOutFile,
RunTimeFile:跑过的账期记录
""" 
def tfWrite(cut_dataset,tfOutFile,runtime,mergeWeek=6):
    print "build dic.."
    dictionary = corpora.Dictionary(cut_dataset)#用于生成字典类似与table，Counter模块中count
    corpus = [dictionary.doc2bow(text) for text in cut_dataset]#doc2bow将文本转换成稀疏向量
    tfidf = models.TfidfModel(corpus)
        
    #写该账期的词频文件
    tff=open(tfOutFile.split(".")[0]+"_"+runtime+".txt","w")
    tfdict=sorted(tfidf.dfs.iteritems(),key=lambda x:x[1],reverse=True)
    for listele in tfdict:
        out=dictionary[listele[0]].strip()+" "+str(listele[1])
        tff.writelines(out+"\n")
    tff.close()
    #写总阶段的词频文件
    mondays=[str(runtime)]#记录每个星期的第一天
    t = time.strptime(str(runtime), "%Y%m%d")
    y,m,d = t[0:3]
    for midx in range(mergeWeek):
        #取前一周的时间
        mon=(datetime.datetime(y,m,d) - datetime.timedelta(days = 7+midx*7)).strftime("%Y%m%d")
        mondays.append(mon)
    print mondays
    #合并词频文件
    mergeWeek=0
    for monday in mondays:
        tfpath=tfOutFile.split(".")[0]+"_"+monday+".txt"#历史词频文件
        if os.path.exists(tfpath):
            mergeWeek=mergeWeek+1
    tf_dic={}#历史词频文件合并
    for monday in mondays:
        tfpath=tfOutFile.split(".")[0]+"_"+monday+".txt"#历史词频文件
        if os.path.exists(tfpath):
            print "********************include "+tfpath+"into"+tfOutFile+"********************"
            mlines=open(tfpath,"r").readlines()
            for line in mlines:
                word=line.split(" ")[0]
                freq=line.split(" ")[1]
                if tf_dic.has_key(word):
                    tf_dic[word]=int(freq)+tf_dic[word]
                else:
                    tf_dic[word]=int(freq)
                    
                    
    tff=open(tfOutFile,"w")
    tf_dic_list=sorted(tf_dic.iteritems(),key=lambda x:x[1],reverse=True)
    for ele in tf_dic_list:
        out=ele[0]+" "+str(np.ceil(ele[1]))
        tff.writelines(out+"\n")
    tff.close()
        
    return tf_dic
    
#返回需要重新添加标签的样本
"""
seldom_word_thea: #选择最小单词频率，作为词频文件中少见词的词频阈值
score_thea=0.07:句子常见程度下限
"""
def needLabelWrite(dataset,cut_dataset,tf_dict,labelOutPath,runtime,seldom_word_min=2,score_thea=0.07):
    dictionary = corpora.Dictionary(cut_dataset)#用于生成字典类似与table，Counter模块中count
    bow_corpus = [dictionary.doc2bow(text) for text in cut_dataset]#doc2bow将文本转换成稀疏向量
  
    usual_word_score=[]#记录每篇文章中常见词的比重
    for article in bow_corpus:
        seldom_word_count=0
        article_len= 0
        for word in article:
            word_freqency=tf_dict[dictionary[word[0]].encode("utf-8")]
            article_len=article_len+1
            if word_freqency<seldom_word_min:
                seldom_word_count=seldom_word_count+1
        if article_len==0:
            usual_word_score.append(1.0)
            continue
        usual_word_score.append(1-seldom_word_count/float(article_len))
    
    #返回需要重新添加标签的样本
    needLabel=[]
    qq=[]
    for sidx,score in enumerate(usual_word_score):
        if (score<score_thea):
            qq.append(str(sidx)+"^"+dataset[sidx])
            needLabel.append(dataset[sidx])
    f0=open(labelOutPath,"w")
    if ctg=="comt":
        grp="评论"
    elif ctg=="news":
        grp="新闻"
    for ele in needLabel:
        ele=ele.replace("^","-")
        f0.write(runtime+"^"+ele+"^"+grp+"\n")#grp文本的组别
    f0.close()
    
    return usual_word_score
    

    

if __name__=="__main__":
    """
    自动删选一些生僻文本，利用词频信息。把这些生僻的文本作为需要重新添加标签的样本
    seldom_word_min=2 #选择最小单词频率，作为词频文件中少见词的词频阈值
    score_thea=0.7#句子常见度下限，句子常见度指常见词在整句中所占有的比例
    挑选出常见度低的作为生僻文本
    """
#    articlesPath=sys.argv[1]#所有未标记的文本,[文件格式：一行一篇文本]
#    tfOutFile=sys.argv[2]#tf词频信息文件
#    labelOutPath=sys.argv[3]#待标记的文本
#    ctg=sys.argv[4]#文本的组别，评论或者新闻
#    runtime=sys.argv[5]#脚本执行的账期，会从数组库中读取一周的文本数据
#    mergeWeek=6#合并几周的词频信息
#    if len(sys.argv)==7:
#        mergeWeek=sys.argv[6]


    articlesPath=r"D:/data/comment/server/comtUnlabeledSample.txt"
    tfOutFile="D:/data/comment/comttf.txt"
    labelOutPath="D:/data/comment/SougouNeedLable.txt"
    ctg="comt"
    runtime="20161215"
    mergeWeek=6
    
    dataset,cut_dataset=readCorpus(articlesPath,lineno=100)
    tf_dict=tfWrite(cut_dataset,tfOutFile,runtime,mergeWeek)   
    seldom_word_min=2 #选择最小单词频率，作为词频文件中少见词的词频阈值
    score_thea=0.7#句子常见程度下限，超过此值的作为以训练过得样本
    usual_word_score=needLabelWrite(dataset,cut_dataset,tf_dict,labelOutPath,runtime,score_thea=score_thea)


