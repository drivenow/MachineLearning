# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:48:53 2017

@author: Administrator
"""
"""
使sgd算法中的步长呈指数下降.
1.通过callbacks中调用LearningRateScheduler
2.decay中指定decay
"""
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import math

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
#model.fit(train_set_x, train_set_y, validation_split=0.1, nb_epoch=200, batch_size=256, callbacks=[lrate])