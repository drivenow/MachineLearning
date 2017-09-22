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

disk='J:/'
n_dim=100
#建立正则表达式
strinfo = re.compile("\\.|。|，|,|\"|“|”|‘|’|；|：|！|、|| |《|》|<|>|…|:|;|？|（|）|(|)")
#加载word2vec模型
#==============================================================================
# imdb_w2v = word2vec.Word2Vec.load(disk+u"script/gensim/tourNewsClfy/word2vec.Model/newsAll4G.model.vec50")
#==============================================================================
#imdb_w2v = word2vec.Word2Vec.load(r'D:/data/news/model/newsTitle.all.100')

#文本Array
textArray=[]
#类别Array
contClfyArray=[]
#情感类别
emotClfyArray=[]

fileRes=open(u"D:/data/news/新闻样本why.txt")
for fileReader in fileRes:
    content=fileReader.strip()
    contentSplit=content.split("^")
    contentWSplit=" ".join(jieba.cut(strinfo.sub('',contentSplit[0]), cut_all=False))
    textArray.append(contentWSplit)
    contClfyArray.append(oneHotClfyCode('cont','encode',contentSplit[1]))
    emotClfyArray.append(oneHotClfyCode('emot','encode',contentSplit[2]))
fileRes.close


    
#==============================================================================
# textNpArray=np.concatenate([buildWordVector(z, n_dim) for z in textArray])
#==============================================================================
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

from sklearn.ensemble import RandomForestClassifier

#随机森林训练内容分类
clfCont=RandomForestClassifier(n_estimators=200)
clfCont.fit(xTrainCont, yTrainCont)
scoreCont = clfCont.score(xTestCont,yTestCont)
print scoreCont
#==============================================================================
# print '各feature的重要性：%s' % clfCont.feature_importances_
#==============================================================================

#随机森林训练情感分类
clfEmot=RandomForestClassifier(n_estimators=200)
clfEmot.fit(xTrainEmot, yTrainEmot)
scoreEmot = clfEmot.score(xTestEmot,yTestEmot)
print scoreEmot
#==============================================================================
# print '各feature的重要性：%s' % clfCont.feature_importances_
#==============================================================================











