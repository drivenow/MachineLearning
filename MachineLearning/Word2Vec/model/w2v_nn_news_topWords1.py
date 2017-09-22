# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
from gensim.models import word2vec
import jieba
import re
import numpy as np
from sklearn.cross_validation import train_test_split

def oneHotClfyCode(typ,fun,clfy):
    if typ=='emot':
        if fun=='encode':
            if clfy==r'好':
                return [1,0,0]
            elif clfy==r'中':
                return [0,1,0]
            elif clfy==r'差':
                return [0,0,1]
        elif fun=='decode':
            if sum(abs(clfy-[1,0,0]))==0:
                return r'好'
            elif sum(abs(clfy-[0,1,0]))==0:
                return r'中'
            elif sum(abs(clfy-[0,0,1]))==0:
                return r'差'
    elif typ=='cont':
        if fun=='encode':
            if clfy==r'动向公告':
                return [1,0,0,0,0,0]
            elif clfy==r'发展推广':
                return [0,1,0,0,0,0]
            elif clfy==r'旅游乱象':
                return [0,0,1,0,0,0]
            elif clfy==r'市场监管':
                return [0,0,0,1,0,0]
            elif clfy==r'游记攻略':
                return [0,0,0,0,1,0]
            elif clfy==r'其他':
                return [0,0,0,0,0,1]
        elif fun=='decode':
            if sum(abs(clfy-[1,0,0,0,0,0]))==0:
                return r'动向公告'
            elif sum(abs(clfy-[0,1,0,0,0,0]))==0:
                return r'发展推广'
            elif sum(abs(clfy-[0,0,1,0,0,0]))==0:
                return r'旅游乱象'
            elif sum(abs(clfy-[0,0,0,1,0,0]))==0:
                return r'市场监管'
            elif sum(abs(clfy-[0,0,0,0,1,0]))==0:
                return r'游记攻略'
            elif sum(abs(clfy-[0,0,0,0,0,1]))==0:
                return r'其他'

def oneHotClfyCodeArray(typ,fun,npArray):
    npArrayCode=[]
    for array in npArray:
        code=oneHotClfyCode(typ,fun,array)
        npArrayCode.append(code)
    return np.array(npArrayCode)

#每句话的中各个词特征值的均值
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
#==========================================================================

#disk='J:/'
#n_dim=100
##建立正则表达式
#strinfo = re.compile("\\.|。|，|,|\"|“|”|‘|’|；|：|！|、|| |《|》|<|>|…|:|;|？|（|）|(|)")
##加载word2vec模型
##==============================================================================
## imdb_w2v = word2vec.Word2Vec.load(disk+u"script/gensim/tourNewsClfy/word2vec.Model/newsAll4G.model.vec50")
##==============================================================================
#imdb_w2v = word2vec.Word2Vec.load(r'D:/data/news/model/newsTitle.all.100')
#
##文本Array
#textArray=[]
##类别Array
#contClfyArray=[]
##情感类别
#emotClfyArray=[]
#
#
#fileRes=open(u"D:/data/news/新闻样本.txt")
#for fileReader in fileRes:
#    content=fileReader.strip()
#    contentSplit=content.split("^")
#    contentWSplit=" ".join(jieba.cut(strinfo.sub('',contentSplit[0]), cut_all=False))
#    textArray.append(contentWSplit)
#    contClfyArray.append(oneHotClfyCode('cont','encode',contentSplit[1]))
#    emotClfyArray.append(oneHotClfyCode('emot','encode',contentSplit[2]))
#fileRes.close
#%%
n_dim=100


def getStopWords(stopWordsPath):
    stopWords=[]
    fileRes=open(stopWordsPath)
    for fileReader in fileRes:
        stopWords.append(fileReader.strip().decode('utf-8'))
    return stopWords
#==============================================================================
#     return {}.fromkeys([x[0] for x in stopWords])     
#==============================================================================

print u'loading stopWords...'
#==============================================================================
# stopWordsPath=disk+u"script/集团云公司项目/新闻分类/分词数据/stopWords.txt"
#==============================================================================
stopWordsPath=u"D:/data/news/stop.txt"
stopWords=getStopWords(stopWordsPath)

print u'split words...'
def WordSplit(stopWords,text):
    wspText=''
    wsps=jieba.cut(text, cut_all=False,HMM=True)
    for wsp in wsps:
        if wsp not in stopWords:
            wspText =wspText+wsp+' '
    return wspText.strip()

#==============================================================================
# fileRes=open(disk+u"script/集团云公司项目/新闻分类/dataSet/新闻样本.txt")
#==============================================================================
fileRes=open(u"D:/data/news/标题内容标签.txt")
#fileRes=open(u"D:/data/news/新闻标题.txt")
#文本Array
textArray=[]
#类别Array
contClfyArray=[]
#情感类别
emotClfyArray=[]


#建立正则表达式
strinfo = re.compile("\\.|。|，|,|\"|“|”|‘|’|；|：|！|、|    |《|》|<|>|…|:|;|？|（|）|(|)")

for fileReader in fileRes:
    content=fileReader.strip()
    contentSplit=content.split("^")
#==============================================================================
#     contentWSplit=" ".join(jieba.cut(contentSplit[0], cut_all=False,HMM=True))
#==============================================================================
    contentWSplit=WordSplit(stopWords,contentSplit[0]+contentSplit[1])
    contentWSplit=strinfo.sub('',contentWSplit)
    textArray.append(contentWSplit)
    contClfyArray.append(oneHotClfyCode('cont','encode',strinfo.sub('',contentSplit[2])))
    emotClfyArray.append(oneHotClfyCode('emot','encode',strinfo.sub('',contentSplit[3])))
fileRes.close


print u'create tfidf...'
#==============================================================================
# 构建IDF词典
#==============================================================================
from gensim import corpora
from gensim.models import TfidfModel
textArraySplit=[line.split() for line in textArray]
dictionary = corpora.Dictionary(textArraySplit)#用于生成字典
corpus = [dictionary.doc2bow(text) for text in textArraySplit]#doc2bow将文本转换成稀疏向量
tfidf = TfidfModel(corpus)

def saveIdf(tfidfModel,idfOutFile):
    idff=open(idfOutFile,"w")
    idfdict=sorted(tfidf.idfs.iteritems(),key=lambda x:x[1],reverse=True)
    for lidx,idfTuple in enumerate(idfdict):
        print lidx,idfTuple
       
        out=dictionary[idfTuple[0]].strip().encode("utf-8")+" "+str(idfTuple[1])+"\n" 
        idff.write(out)
    idff.close()
    
idfOutFile="D:/data/news/tfidf/titleIdf.txt"
saveIdf(tfidf,idfOutFile)
#%% 每篇文本提取关键词

"""
加入idf词信息，提取文章的关键词
"""
import os
import jieba.analyse
def keyWordExt(content,top,withWeightEn,rate):
    tags = jieba.analyse.extract_tags(content, topK=top, withWeight=withWeightEn)
    tagCnt=0
    keyWord=[]
    baseVal=0.1
    for tag in tags:
        if tagCnt==1:
            keyWord.append(tag[0].encode("utf-8"))
            baseVal=tag[1]
        else:
            if tag[1]>=baseVal*rate :
                keyWord.append(tag[0].encode("utf-8"))
        tagCnt+=1
    return keyWord
    
#set idf path
textExtractArray=[]
idfPath="D:/data/news/tfidf/titleIdf.txt"
if  os.path.exists(idfPath):
    jieba.analyse.set_idf_path(idfPath)
    
for line in textArray:
    line="".join(line)
    if line == None:
        continue
    keyWords=keyWordExt(line,5,True,0.3)
    textExtractArray.append(keyWords)
    
textArray=textExtractArray



#%%   
#==============================================================================
# textNpArray=np.concatenate([buildWordVector(z, n_dim) for z in textArray])
#==============================================================================
a="a"

contClfyNpArray=np.array(contClfyArray)
emotClfyNpArray=np.array(emotClfyArray)

xTrainContText, xTestConText, yTrainCont, yTestCont = train_test_split(textArray, contClfyNpArray, test_size=0.2)
xTrainEmotText, xTestEmotText, yTrainEmot, yTestEmot = train_test_split(textArray, emotClfyNpArray, test_size=0.2)

xTrainCont=np.concatenate([buildWordVector(z, n_dim) for z in xTrainContText])
xTestCont=np.concatenate([buildWordVector(z, n_dim) for z in xTestConText])
xTrainEmot=np.concatenate([buildWordVector(z, n_dim) for z in xTrainEmotText])
xTestEmot=np.concatenate([buildWordVector(z, n_dim) for z in xTestEmotText])

def kerasOutFormat(npArray):
    npArrayFormat=[]
    pbly=[]
    for array in npArray:
        size,=array.shape
        vec = np.zeros(size).reshape((1, size))
        maxPos=np.argmax(array)
        vec[0][maxPos]=1
        npArrayFormat.append(vec[0])
        pbly.append(array[maxPos])
    return np.array(npArrayFormat).reshape(-1,size),pbly
    
def getKerasModel(inputDim,outputDim):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation

    model = Sequential()

    model.add(Dense(128, input_dim=inputDim,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(outputDim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


#神经网络训练内容分类
modelCont=getKerasModel(n_dim,6)
#==============================================================================
# modelCont.fit(textNpArray, contClfyNpArray, batch_size=64, nb_epoch=20,
#    verbose=1,validation_split=0.2,shuffle=True)
#==============================================================================
modelCont.fit(xTrainCont, yTrainCont, batch_size=256, nb_epoch=200,
   verbose=1,validation_data=(xTestCont,yTestCont),shuffle=True)
contClfyPredict=modelCont.predict(xTestCont)
contClfyPredictFormat,contpbly=kerasOutFormat(contClfyPredict)
contClfyPredictDecode=oneHotClfyCodeArray('cont','decode',contClfyPredictFormat)

#神经网络训练情感分类
modelEmot=getKerasModel(n_dim,3)
#==============================================================================
# modelEmot.fit(textNpArray, emotClfyNpArray, batch_size=64, nb_epoch=20,
#    verbose=1,validation_split=0.2,shuffle=True)
#==============================================================================
modelEmot.fit(xTrainEmot, yTrainEmot, batch_size=256, nb_epoch=200,
   verbose=1,validation_data=(xTestEmot,yTestEmot),shuffle=True)
emotClfyPredict=modelEmot.predict(xTestEmot)
emotClfyPredictFormat,emotpbly=kerasOutFormat(emotClfyPredict)
emotClfyPredictDecode=oneHotClfyCodeArray('emot','decode',emotClfyPredictFormat)













