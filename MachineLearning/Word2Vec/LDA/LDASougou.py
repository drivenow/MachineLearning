#encoding:utf-8
'''
Created on 2015年10月25日

@author: Administrator
'''
#66.97

from gensim import corpora, models, similarities
import jieba.posseg as pseg
import os
import DocSplit
import time
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

basePath = r"D:/data/SougouNews/news/";
#%%  jieba分词
#outBasePath=r"D:/data/SougouNews/newscut/"
#fileList = os.listdir(basePath)
#dataset = []
#n = 0
#start=time.clock()
#for file in fileList:
#    if(n<500000):
#        print n
#        if os.path.isfile(basePath+file)==True:
#            line = open(basePath+file,"r").readline()
#            line=line.encode("utf-8")
#            line1=[]
#            tmp=pseg.cut(line)
#            for w in tmp:
#                ab=w.flag.encode("utf-8")
#                if ((ab == 'n'or ab == 'v' or ab == 'a') and len(w.word.encode("utf-8"))>1): 
#                    line1.append(w.word.encode("utf-8"))
#                    of=open(outBasePath+str(n).zfill(6)+".txt","w")
#                    for word in line1:
#                        of.write(word+"\n")
#                    of.close()
#            dataset.append(line1)
#        if (n%1000==0):
#            print n
#        n=n+1
#end=time.clock()
#print " %f s" % (end - start)
#%% 读取数据集
basePath=r"D:/data/SougouNews/newscut/"
fileList = os.listdir(basePath)
dataset = []
n = 0
start=time.clock()
for file in fileList:
    if(n<200000):
        if (n%1000==0):
            print n
        if os.path.isfile(basePath+file)==True:
            lines = open(basePath+file,"r").readlines()
            dataset.append(lines)
        n=n+1
end=time.clock()
print " %f s" % (end - start)

#%%
#%% tfidf加权  
print "build dic.."
start=time.clock()
dictionary = corpora.Dictionary(dataset)#用于生成字典类似与table，Counter模块中count
end=time.clock()
print " %f s" % (end - start)
    #print dictionary.token2id  

corpus = [dictionary.doc2bow(text) for text in dataset]#doc2bow将文本转换成稀疏向量

#%% 
print "tf-idf..."
start=time.clock()
tfidf = models.TfidfModel(corpus)  
# print tfidf.dfsx  
# print tfidf.idf  
corpus_tfidf = tfidf[corpus]
end=time.clock()
print " %f s" % (end - start)

#%% 
"""主题模型lda，可用于降维 ,lda流式数据建模计算，chunck，提取50个主题  
id2word
num_topics
update_every
chunksize：文档大小
passes
"""
print "lda..."
start = time.clock()
lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=1)  
end=time.clock()
print " %f s" % (end - start)

#%% 测试
def getTerm(terms,dictionary):
    out=[]
    for term in terms:
        out.append(dictionary[term[0]])
    return out
#获取相似词

"""
For num_topics number of topics, return num_words most significant words (10 words per topic, by default).
log：将概率对数化
注：不同于LSA,topic之间并没有明显的主题顺序（即不能提取前n个主题）,两次LDA训练的结果也会导致不同的topic顺序
"""
topics_dictionary=lda.show_topics(num_topics=10,num_words=10,log=False)#打印出模型的每个主题（id）和频率
tmp=[]
for i in topics_dictionary:
    tmp.append(i)
#%% #提取前面10个主题
top_topic=lda.top_topics(dataset,num_words=10)#caculate Umass topic coherence?
for i in top_topic:
    print i
#提取前面10个主题
for i in range(0,10):  
    print lda.print_topics(i)[0][1]

##获取每个主题下前几个单词
term_in_topic=lda.get_topic_terms(topicid=5,topn=10)
term_in_topic_out=getTerm(term_in_topic,dictionary)


doc_lda = lda[corpus_tfidf]#利用原模型预测新文本主题,每篇给出主题预测 
term_in_topic_out1=getTerm(doc_lda,dictionary)
#topics_doc=lda.get_document_topics(dataset2[0])#利用原模型预测新文本主题 
 


