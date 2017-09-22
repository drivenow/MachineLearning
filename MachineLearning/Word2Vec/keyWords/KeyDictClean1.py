# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:00:52 2017
@author: Administrator
"""

"""
从文件中获取关键词，文件格式：单词\t单词...
"""
def getKeywordDict(fpath,delimeter="\t"):
    fin=open(fpath,"r")
    keywordDict={}
    for line in fin:
#        print line
        line=line.strip()
        if line=="":continue
        line=line.replace("\n","")
        wordsList=line.split(delimeter)
        keywordDict=keyWordUnique(keywordDict,wordsList)
    fin.close()
    return keywordDict
    

"""
    #数据去重
    将在wordList中，不在keyWordDict中的词，加入到keyWordDict
"""    
def keyWordUnique(keyWordDict,wordsList):
    for word in wordsList:
        if keyWordDict.has_key(word):
            continue
        else:
            keyWordDict[word]=0
    return keyWordDict


"""
根据强相关，把类A的词作为类A独有的,将其他类中的该词删掉
strongWordsDict={ctg:{word:0}}
"""
def strongFilter(ctgKeywordDict,strongWordsDict):
    for strongCtg in strongWordsDict.keys():
        for word in strongWordsDict[strongCtg].keys():
            for otherCtg in ctgKeywordDict.keys():
                if strongCtg==otherCtg:
                    #判断strongword是否在其中，否加入
                    if ctgKeywordDict[otherCtg].has_key(word)==False:
                        print "add strong word :"+word
                        ctgKeywordDict[otherCtg][word]=0
                        continue
                else:
                    #判断strongword是否在其中，是的话删除
                    if ctgKeywordDict[otherCtg].has_key(word)==True:
                        print "pop strong word :"+word
                        ctgKeywordDict[otherCtg].pop(word)
                        continue
    return ctgKeywordDict
    
"""
根据弱相关，将类A中弱相关的词去掉
"""
def weakFilter(ctgKeywordDict,weakWordsDict):
    for weakCtg in weakWordsDict.keys():
        for word in weakWordsDict[weakCtg].keys():
            for otherCtg in ctgKeywordDict.keys():
                if weakCtg!=otherCtg:
                    continue#只去掉所属类的弱相关词
                else:
                    if ctgKeywordDict[otherCtg].has_key(word)==False:continue
                    print "pop weak word :"+word
                    ctgKeywordDict[otherCtg].pop(word)
    return ctgKeywordDict


def writeCtgDict(ctgDict,fpath):
    outf=open(fpath,"w")
    for word in ctgDict.keys():
        outf.write(word+"\t")
    outf.close()                 
    
    

if __name__=="__main__":
    """
    对词库进行清理。
    根据强相关，把类A的词作为类A独有的,将其他类中的该词删掉
    根据弱相关，将类A中弱相关的词去掉
    """
    fpath1=u"D:/data/news/key/merge/unique旅游乱象merge.txt"
    fpath2=u"D:/data/news/key/merge/unique游记攻略merge.txt"
    fpath3=u"D:/data/news/key/merge/unique旅游动态merge.txt"
    fpath4=u"D:/data/news/key/merge/unique市场监督merge.txt"
    fpath5=u"D:/data/news/key/merge/unique其他merge.txt"
    
    strongPath3=u"D:/data/news/key/相关/强相关（旅游动态）.txt"
    
    weakPath1=u"D:/data/news/key/相关/弱相关（旅游动态）-旅游乱象.txt"
    weakPath2=u"D:/data/news/key/相关/弱相关（旅游动态）-游记攻略.txt"
    weakPath4=u"D:/data/news/key/相关/弱相关（旅游动态）-市场监督.txt"
    weakPath5=u"D:/data/news/key/相关/弱相关（旅游动态）-其他.txt"
    
    
    ctgKeywordDict={}#每类包含哪些关键词
    ctgKeywordDict["1"]=getKeywordDict(fpath1)
    ctgKeywordDict["2"]=getKeywordDict(fpath2)
    ctgKeywordDict["3"]=getKeywordDict(fpath3)
    ctgKeywordDict["4"]=getKeywordDict(fpath4)
    ctgKeywordDict["5"]=getKeywordDict(fpath5)
    
    strongWordsDict={}
    strongWordsDict["3"]=getKeywordDict(strongPath3," ")
    
    ctgKeywordDict=strongFilter(ctgKeywordDict,strongWordsDict)
    
    weakWordsDict={}
    weakWordsDict["1"]=getKeywordDict(weakPath1," ")
    weakWordsDict["2"]=getKeywordDict(weakPath2," ")
    weakWordsDict["4"]=getKeywordDict(weakPath4," ")
    weakWordsDict["5"]=getKeywordDict(weakPath5," ")
    
    ctgKeywordDict=weakFilter(ctgKeywordDict,weakWordsDict)
    
    #%% 将关键词写入文件
    writeCtgDict(ctgKeywordDict["1"],fpath1)
    writeCtgDict(ctgKeywordDict["2"],fpath2)
    writeCtgDict(ctgKeywordDict["3"],fpath3)
    writeCtgDict(ctgKeywordDict["4"],fpath4)
    writeCtgDict(ctgKeywordDict["5"],fpath5)
    
    
    
    
    
    
    