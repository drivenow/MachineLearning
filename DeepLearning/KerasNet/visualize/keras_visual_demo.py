# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:34:06 2016
@author: Administrator
加载训练完成的验证码识别模型（四个位置）,对中间层的输出结果可视化
"""
import sys
sys.path.append(u"D:\project\VertCode\mycode")
import numpy as np
from ImagePre import getPicArrWithLab

def oneHotLab2str(labArr):
    num=[]
    strs=[]
    for i in labArr:
        for idx,j in enumerate(i):
            if np.round(j)==1:
                num.append(idx)
                break
    for i in num:
        if 0<=i<10:
            strs.append(str(i))
        if 9<i<37:
            strs.append(chr(i+97-10))
    return strs

picBasePath="D:/project/VertCode/tuniu"
labPath="D:/project/VertCode/1(1).txt"
X_train,Y_train=getPicArrWithLab(picBasePath,labPath,reverse=True,binary=False,skele=False)

#%% 加载模型
from keras.models import model_from_json
modelPath="D:/project/VertCode/mycode/output/model"
model0 = model_from_json(open(modelPath+"/model0.json").read())    
model0.load_weights(modelPath+"/model0_weight.h5")    
model1 = model_from_json(open(modelPath+"/model1.json").read())    
model1.load_weights(modelPath+"/model1_weight.h5")   
model2 = model_from_json(open(modelPath+"/model2.json").read())    
model2.load_weights(modelPath+"/model2_weight.h5")    
model3 = model_from_json(open(modelPath+"/model3.json").read())    
model3.load_weights(modelPath+"/model3_weight.h5")     

#%%模型可视化
from keras.utils.visualize_util import plot
from keras.models import Model
from keras import backend as K#等价import theano

#args
input_img=X_train[100,:,:,:].reshape(1,1,30,80)
"""
（1）将模型输出到文件
show_shapes：指定是否显示输出数据的形状，默认为False
show_layer_names:指定是否显示层名称,默认为True
"""
plot(model0, to_file='output/model0.png',show_shapes=True)
model0.summary()

"""
(2)新建一个Model,指定input和output.(利用model0.get_layer函数,获得指定层）
"""
layer_name = 'convolution2d_2'
input_layer2=model0.input
output_layer2=model0.get_layer(layer_name).output
intermediate_layer_model = Model(input=input_layer2,output=output_layer2)
intermediate_output2 = intermediate_layer_model.predict(input_img)

"""
encoder = Model(x, z_mean)
"""

#%% 有错误
#"""
#（3）利用后端新建model，Theano或TensorFlow(利用model0.layers函数,获得指定层),别忘记中括号
#1.注意，如果你的模型在训练和测试两种模式下不完全一致，例如你的模型中含有Dropout层，批规范化（BatchNormalization）层等组件，
#你需要在函数中传递一个learning_phase的标记，像这样：
#2.K.learning_phase()不加表示输入输出模型完全一样：
#    K.function([model0.layers[0].input],[model0.layers[3].output])
#"""
#input_layer3=model0.layers[0].input
#output_layer3=model0.layers[3].output
#get_3rd_layer_output = K.function([input_layer3, K.learning_phase()],
#                                  [output_layer3])
## output in test mode = 0
#layer_output3 = get_3rd_layer_output([input_img, 0])[0]
## output in train mode = 1
#layer_output3 = get_3rd_layer_output([input_img, 1])[0]


#%% 显示图像
from matplotlib import pyplot as plt

#Args
output_img=intermediate_output2
sub_row=6#子图的行数
sub_col=6#子图的列数
sub_i=0#子图的编号

for feature_maps in output_img:
    for img in feature_maps:
        sub_i=sub_i+1
        plt.subplot(sub_row,sub_col,sub_i)
        plt.imshow(img)




