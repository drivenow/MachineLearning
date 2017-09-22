# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:02:18 2016

@author: Administrator
"""

from gensim.models import word2vec
import sys
reload(sys)
sys.path.append("D:\OneDrive\codes\python\Word2Vec\util")
sys.setdefaultencoding('utf-8')
import DocSplit
#训练模型

"""
    Word2Vec:无监督训练，Google提供测评方式“A之于B相当于C至于D”, 在神经网络中学习将word映射成连续（高维）向量， 即词向量的构建。
    min_count:default(5),少于min_count次数的单词会被丢弃掉
    size:default（100），隐层单元数
    worker:并行单元，安装Cpython后才有效
    sg:default(0)CBOW训练，SG=1，SKIP_GRAM算法
    sorted_vocab：(default)1,字典默认排序
    
    参数存储：内存中有三个这样的矩阵, 如果你的输入包含100,000个单词, 隐层单元数为200, 则需要的内存大小为100,000 * 200 * 4 * 3 bytes, 约为229MB.
            要一些内存来存储字典huffman树, 但是除非你的单词是特别长的字符串, 大部分内存占用都来自前面说的三个矩阵. 
    语料来一个处理一个：
        sentences = LineSentence('myfile.txt'):一行一句话; 以空白分词的语料.
        Text8Corpus = Iterate over sentences from the “text8” corpus
        BrownCorpus = Iterate over sentences from the Brown corpus
    词模型加减：国王-男+女=女王，词向量可以相加减
        文档中词向量直接取平均（目前没发现加权，而且相似度，是两个词向量相乘结果）
"""

"""
build_vocab:收集单词及其词频来够爱走一个内部字典树结构,若Word2Vec参数中有corpus，自动构建字典
train:
"""

basePath = r"D:\data\SougouNews";
f1=open(basePath+r"\resultbig.txt",'r+')
f2 =  open(basePath+r"\resultbig1.txt",'r+'); 
f3 =  open(basePath+r"\resultbig2.txt",'r+'); 

corpus=f1.readlines()
corpus=DocSplit.jiebaText(corpus)

lines2=f2.readlines()
corpus2=DocSplit.jiebaText(lines2)

lines3=f3.readlines()
corpus3=DocSplit.jiebaText(lines3)

f1.close()
f2.close()
f3.close()

#%% 模型构建
model = word2vec.Word2Vec(min_count=1)  # 训练skip-gram模型; 默认window=5

#model = word2vec.Word2Vec(sentence, size=100, window=5, min_count=1, workers=4)
model.build_vocab(corpus2)

#模型（增量）训练
model.train(corpus2,total_examples=len(corpus2),total_words=4147)
#model.up
#model.train(corpus2)
#model.train(corpus3)

#model.init_sims(replace=True)#model =no more updates
#model.finalize_vocab(update=False)#Build tables and model weights based on final vocabulary settings.

#%% 模型测试
"""
model["税"]获取词向量
most_similar:
    positive:
    negative:
    topn:
doesnt_match:
similarity:          
"""

smcl=model.most_similar(u"银行",topn=5)
for str in smcl:
    print str[0]

smcl = model.most_similar(u"规划", topn=5)
for str in smcl:
    print str[0]

#model.similarity(u"文化",u"旅游")    

model[u"银行"]# raw NumPy vector of a word,维数为size

 
#%% 存储加载模型
#model.save(bashPath+r'/tmp/mymodel')
#new_model = gensim.models.Word2Vec.load(bashPath+r'/tmp/mymodel')
##直接加载由C生成的模型:
#model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
# # using gzipped/bz2 input works too, no need to unzip:
#model=Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
