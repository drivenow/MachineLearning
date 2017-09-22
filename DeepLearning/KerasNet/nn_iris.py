# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 14:06:47 2016

@author: Administrator
"""

from keras.models import Sequential,Model
from keras.layers import Dense,Input,merge
from keras.activations import softmax
from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedShuffleSplit
from keras.callbacks import EarlyStopping
import numpy as np

#lab转换成one-hot编码
def datasetSplit(data,lab):
    ssp = StratifiedShuffleSplit(lab,n_iter=1,test_size=0.20,random_state=1)
    for trainlab,testlab in ssp:
        print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
    X_train=data[trainlab]
    X_test=data[testlab]
    Y_train=np_utils.to_categorical(lab[trainlab],3)
    Y_test=np_utils.to_categorical(lab[testlab],3)
    return X_train,X_test,Y_train,Y_test

nb_epoch=500

input1=Input(shape=(4,))#输入特征维度4
classify_output=3



#%% 一层一个节点
x=Dense(4)(input1)#30,0.33;50,0.85,100,0.36,500,0.96(收敛)
x=Dense(4)(x)#50,0.03，100,0.70
x=Dense(4)(x)#50,0.71,；50,0.49；50,0.86；50.0.88；50,0.32,；100(未收敛)，500（收敛）
output1=Dense(3,activation='softmax')(x)

#%% 一层多个节点
dense1=Dense(4)
x1=dense1(input1)
x2=dense1(input1)
x3=dense1(input1)
assert dense1.get_output_at(0) == x1,"dense1层的第一个节点的输出不是x1"
assert dense1.get_output_at(1) == x2,"dense1层的第二个节点的输出不是x2"
assert dense1.get_output_at(2) == x3,"dense1层的第三个节点的输出不是x3"
merged1=merge([x1,x2,x3],mode='concat')

#%% 模型初始化
model_1point_1layer=Model(input1,output=output1)
model_multiPoint_1layer=Model(input1,output=output1)
"""
optimizer='adadelta',ada自适应的
metrics='accuracy',
loss='categorical_crossentropy'多类对数损失函数
"""
model_1point_1layer.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')
model_multiPoint_1layer.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')

iris=load_iris()
X_train,X_test,Y_train,Y_test=datasetSplit(iris['data'],iris['target'])

#%% 模型训练
"""
monitor='val_loss',需要监视的量
patience=2，监视两相比上一次迭代没有下降，经过几次epoch之后
在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
"""
early_stopping = EarlyStopping(monitor='val_loss', patience=10,verbose=1)
#model_1point_1layer.fit(X_train,Y_train, nb_epoch=nb_epoch,validation_data=(X_test,Y_test))
model_multiPoint_1layer.fit(X_train,Y_train, nb_epoch=nb_epoch,validation_data=(X_test,Y_test),callbacks=[early_stopping])