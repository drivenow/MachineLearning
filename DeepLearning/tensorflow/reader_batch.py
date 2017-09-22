# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:46:06 2017

@author: Shenjunling
"""

import tensorflow as tf
# 生成一个先入先出队列和一个QueueRunner
filenames = ['data/A.csv', 'data/B.csv', 'data/C.csv']
"""
FIFOQueue '_102_input_producer_9' is closed and has insufficient elements (requested 1, current size 0),迭代次数不够报错
"""
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)#默认无限循环数据集
# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# 定义Decoder
example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
example_batch, label_batch = tf.train.shuffle_batch([example,label], batch_size=5, capacity=200, min_after_dequeue=100, num_threads=2)
#example_batch, label_batch = tf.train.batch([example, label], batch_size=5)

# 运行Graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
    for i in range(10):
        e_val,l_val = sess.run([example_batch, label_batch])
#        e_val,l_val = sess.run([example, label])
        print (e_val,l_val)  

    coord.request_stop()
    coord.join(threads)
#    
#%% 迭代控制，多个reader，多个样本
#import tensorflow as tf
#filenames = ['data/A.csv', 'data/B.csv', 'data/C.csv']
#filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=3)  #num_epochs重复遍历数据集多少次
#reader = tf.TextLineReader()
#key, value = reader.read(filename_queue)
#record_defaults = [['null'], ['null']]
#example_list = [tf.decode_csv(value, record_defaults=record_defaults)
#                  for _ in range(2)]
#example_batch, label_batch = tf.train.batch_join(
#      example_list, batch_size=5)
#
#
#init_local_op = tf.initialize_local_variables()
#with tf.Session() as sess:
#    sess.run(init_local_op)   # 初始化本地变量，否则报错
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    try:
#        while not coord.should_stop():
#            e_val,l_val = sess.run([example_batch, label_batch])
#            print (e_val,l_val) 
#    except tf.errors.OutOfRangeError:
#        print('Epochs Complete!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
#    coord.request_stop()
#    coord.join(threads)