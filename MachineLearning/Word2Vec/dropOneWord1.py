# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:04:52 2016
@author: Administrator
"""
"""
对搜狗新闻标题，扣去最后一个单词，用剩余单词训练一个词向量。
判断：最相似的两个词向量，最后一个单词是否相同
"""
from gensim.models import word2vec
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("D:/OneDrive/codes/python/Word2Vec/util")
from WordVecUtil import *
from TextPre import *
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence


basePath = r"D:/data/";

newsTitle = open(basePath+"newsTitle.txt","r")
leaveOne=[]
titles=[]#[文本格式]：每一行是分词后的标题

for line in newsTitle.readlines():
    if not line.strip()=="":
        lines=line.split()
        titles.append(lines[:-1])
        leaveOne.append(lines[-1])

#%% 
print "build word2vec model: ..."
modelsize=100
def w2vPre(modelsize):
    model = word2vec.Word2Vec(size=modelsize, window=5, min_count=1, workers=3)
    model.build_vocab(titles)
    model.train(titles)
#==============================================================================
# model= word2vec.Word2Vec.load(r'D:/data/news/model/newsTitle.all.100')
#==============================================================================

title_vec=calcuCorpusVec(model,titles[:100],modelsize,True)#编码一致设为True

#%% 将最相似的结果写入文件
#将最相似的编号对应的文本找出，写入文件 

def writeDropOne(outfile,midx,mdis):       
    outfile=open(outfile,"w")
    outfile.write(list2str(titles[90001])+"\n"+leaveOne[90001].encode("utf-8")+"\n")
    for i in midx:
        toptitle=list2str(titles[int(i)])
        line=toptitle+"\n"+leaveOne[int(i)].encode("utf-8")+"\n"
        outfile.write(line)
    outfile.close()

midx5,mdis5 = mostsim(title_vec[90001],90001,10,title_vec)
outfile=basePath+"result14.txt"
midx=midx5
mdis=mdis5
writeDropOne(outfile,midx,mdis)



    