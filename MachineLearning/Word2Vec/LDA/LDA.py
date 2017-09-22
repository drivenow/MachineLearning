#encoding:utf-8
'''
Created on 2015年10月25日
1. LDA中所谓latent topic不同于我们常说的topic（如体育），没有实际意义上的解释，
有点类似于无指导聚类的类别。通常我们会将每个topic+相对关键词作为新扩展特征后续处理。
2.LDA本身可以看作是聚类，看你的描述是做分类问题。你可以尝试：1）使用有监督的LDA；
3.利用每个词属于每个topic的概率作为该词的特征来构建分类器

@author: Administrator
'''

import nltk  
from gensim import corpora, models, similarities
import DocSplit

basePath = r"D:\data\SougouNews";
f2 =  open(basePath+r"\resultbig1.txt",'r+'); 
f3 =  open(basePath+r"\resultbig2.txt",'r+'); 


lines2=f2.readlines()
dataset2=DocSplit.jiebaText(lines2)#[文本格式]：[[分词后的文本],···,[]]

lines3=f3.readlines()
dataset3=DocSplit.jiebaText(lines3)

f2.close()
f3.close()

#%% tfidf加权  
dictionary = corpora.Dictionary(dataset3)#用于生成字典类似与table，Counter模块中count
#print dictionary.token2id  #字典{词：词编号）
corpus = [dictionary.doc2bow(text) for text in dataset3]#doc2bow将文本转换成稀疏向量
tfidf = models.TfidfModel(corpus)  
# print tfidf.dfsx  
# print tfidf.idf  
corpus_tfidf = tfidf[corpus]

#%% 
"""主题模型lda，提取50个主题  
id2word：
num_topics：主题数
update_every
chunksize：文档大小
passes
支持增量训练
"""
lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10, update_every=1, chunksize=10000, passes=1)
#lda = models.LdaMulticore(workers=4,corpus=corpus_tfidf, id2word=dictionary, num_topics=10, update_every=1, chunksize=10000, passes=1)  
#lda.update(corpus=corpus2)

#%% 测试
"""
lda.print_topics(i)[0]
lda[dataset2]:得到文档的主题分布
"""
"""
根据字典，将单词编号转化成单词
"""
def getTerm(terms,dictionary):
    out=[]
    for term in terms:
        out.append(dictionary[term[0]])
    return out
#获取相似词
print (dictionary[1])
print "相似",

"""
For num_topics number of topics, return num_words most significant words (10 words per topic, by default).
log：将概率对数化
注：不同于LSA,topic之间并没有明显的主题顺序（即不能提取前n个主题）,两次LDA训练的结果也会导致不同的topic顺序
"""
topics_dictionary=lda.show_topics(num_topics=5,num_words=5,log=False)#打印出模型的每个主题（id）和频率
tmp=[]
for i in topics_dictionary:
    tmp.append(i)

#print_topics(num_topics=20, num_words=10)
#show_topics(num_topics=20, num_words=10)
for i in range(0,10):  
    print lda.print_topics(i)[0]  

"""
get_term_topics(word_id, minimum_probability=None):返回word最可能的主题，可以作为word的特征训练
get_topic_terms(topicid, topn=10)#Return a list of (word_id, probability) 2-tuples for the most probable words in topic topicid.

"""
##获取指定主题下前几个单词
term_in_topic=lda.get_topic_terms(topicid=1,topn=10)
term_in_topic_out=getTerm(term_in_topic,dictionary)

#doc_lda = lda[corpus_tfidf]#利用原模型预测新文本主题,每篇给出主题预测 
#getTerm(doc_lda,dictionary)
#topics_doc=lda.get_document_topics(dataset2[0])#利用原模型预测新文本主题 
 
#%%未知
top_topic=lda.top_topics(dataset2,num_words=10)#caculate Umass topic coherence?
for i in top_topic:
    print i


