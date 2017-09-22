# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:33:31 2017

@author: Shenjunling
"""
"""
tf.layers接传统的train_op训练，
但是这里的正则化机制，和batch_normalization的is_train机制怎么自己写？不明确
"""
from tensorflow.contrib import learn
import numpy as np
# Load training and eval data
mnist = learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

import tensorflow as tf

img = tf.placeholder(tf.float32, [None,784])
labels = tf.placeholder(tf.float32,[None])

fc1 = tf.layers.dense(img,1024,activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1,1024,activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2,10)

onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=fc3)

train_op = tf.train.RMSPropOptimizer(0.001).minimize(loss)
prediction = tf.equal(tf.arg_max(tf.nn.softmax(fc3),1), tf.arg_max(onehot_labels,1))

accu = tf.reduce_mean(tf.cast(prediction,  tf.float32))

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    _,ac = sess.run([train_op,accu], feed_dict={img:train_data,labels:train_labels})
    print(ac)

