# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:58:18 2017

@author: Administrator
"""
from keras.callbacks import Callback
import logging
from mykeras.utils.my_generic_utils import MyProgbar


def myLogger(filename,loglevel=logging.INFO):
    """创建日志对象
    级别高低顺序：NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
    　　如果把looger的级别设置为INFO， 那么小于INFO级别的日志都不输出， 大于等于INFO级别的日志都输出　　
    """
    # 创建一个logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(loglevel)
    # 创建一个handler，用于写入日志文件 
    fh = logging.FileHandler(filename) 
    fh.setLevel(loglevel) 
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    return logger


class MyProgbarLogger(Callback):
    """日志写入指定文件
    将每个epoch的结果写入日志
    example:
        myLogger = MyProgbarLogger(to_file=logBasePath+"CNN3d_model.log")
        CNN3d_model.fit(X_train,Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
              validation_data=[X_test,Y_test],callbacks=[myLogger])
    """
    
    def __init__(self,to_file,verbose=1):
         super(MyProgbarLogger, self).__init__()
         self.logger = myLogger(to_file)
        
    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            self.logger.info('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
            self.progbar = MyProgbar(self.logger,target=self.params['nb_sample'],
                                   verbose=self.verbose)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.params['nb_sample']:
            self.log_values = []


    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))


        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        # 取消每次batch训练完写日志
#        if self.verbose and self.seen < self.params['nb_sample']:
#            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)