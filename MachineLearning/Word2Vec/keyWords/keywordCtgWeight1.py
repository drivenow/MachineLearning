# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:54:32 2017
@author: Administrator
"""

"""
#数据去重
    将wordList中首次出现的词，加入到keyWordDict
"""    
def keyWordUnique(keyWordDict,wordsList):
    for word in wordsList:
        if keyWordDict.has_key(word):
            continue
        else:
            keyWordDict[word]=0
    return keyWordDict
    
"""
从文件中获取关键词
"""
def getKeywordDict(fpath):
    fin=open(fpath,"r")
    keywordDict={}
    for line in fin:
        print line
        wordsList=line.split("\t")
        keywordDict=keyWordUnique(keywordDict,wordsList)
    fin.close()
    return keywordDict

"""
找出那些词是被多个类共有的

return: interSectionWords#储存各个类有交集的词{keyword:[incalss]}
"""
def getInterSectionWordsCtgList(ctgKeywordDict):
    interSectionWords={}
    for classi in range(1,len(ctgKeywordDict.keys())+1):
        keywordDicti=ctgKeywordDict[str(classi)]#i类的关键词
        for word in keywordDicti.keys():
            if interSectionWords.has_key(word)==True:continue#不重复计算
            
            flag=False
            inClass=[classi]
            for classj in range(classi+1,len(ctgKeywordDict.keys())+1):
                if ctgKeywordDict[str(classj)].has_key(word):
                    flag=True
                    inClass.append(classj)
            if flag==True:
                interSectionWords[word]=inClass
    return interSectionWords

"""
每类的关键词权重并不相同，每类独有的关键词，权重大，多类共有的关键词，权重小
weight=1/包含该key的类共有多少
return: interSectionWordsDict{key:weight}
"""   
def getInterSectionWordsWeight(interSectionWordsDict):
    interSectionWordsWeightDict={}
    for word in interSectionWordsDict.keys():
        interSectionWordsWeightDict[word]=1.0/len(interSectionWordsDict[word])
    return interSectionWordsWeightDict
    

    
                
                
if __name__=="__main__":
    """
    找出五个类的交叉词
    并赋予权重：1/n
    """
    fpath1=u"D:/data/news/key/merge/unique旅游乱象merge.txt"
    fpath2=u"D:/data/news/key/merge/unique游记攻略merge.txt"
    fpath3=u"D:/data/news/key/merge/unique旅游动态merge.txt"
    fpath4=u"D:/data/news/key/merge/unique市场监督merge.txt"
    fpath5=u"D:/data/news/key/merge/unique其他merge.txt"
    
    ctgKeywordDict={}#每类包含哪些关键词
    ctgKeywordDict["1"]=getKeywordDict(fpath1)
    ctgKeywordDict["2"]=getKeywordDict(fpath2)
    ctgKeywordDict["3"]=getKeywordDict(fpath3)
    ctgKeywordDict["4"]=getKeywordDict(fpath4)
    ctgKeywordDict["5"]=getKeywordDict(fpath5)
    
    
    interSectionWordsCtgDict=getInterSectionWordsCtgList(ctgKeywordDict)
    interSectionWordsWeightDict=getInterSectionWordsWeight(interSectionWordsCtgDict)
    #将关键词权重写入文件
    fout=open(u"D:/data/news/key/merge/交叉词权重.txt","w")
    for word in interSectionWordsWeightDict.keys():
        ctgs=[str(i) for i in interSectionWordsCtgDict[word]]
        outline=word.replace("\n","")+"\t"+('%.2f' % interSectionWordsWeightDict[word])+"\t"+" ".join(ctgs)+"\n"
        fout.write(outline)
    fout.close()
    
    #%%
