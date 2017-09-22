# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(fearures, labels, mode):
    input_layer = tf.reshape(fearures, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer='SGD'
        )
    predictions = {
        'class': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    # dataload
    from keras.datasets import mnist
    (train_dataset, train_labels), (test_dataset, test_labels) = mnist.load_data()
    train_dataset = np.asarray(train_dataset,np.float32)
    train_labels = np.asarray(train_labels,np.float32)
    test_dataset = np.asarray(test_dataset,np.float32)
    test_labels = np.asarray(test_labels,np.float32)
    


    mnist_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/data/mylog/KerasDL/nn_layer_demo'
    )
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    mnist_classifier.fit(x=train_dataset, y=train_labels, batch_size=10, steps=20000, monitors=[logging_hook])

    metrics = {
    'accuray':
        learn.MetricSpec(
            metric_fn=tf.metrics.accuracy,prediction_key='class'
        )
    }

    eval_results = mnist_classifier.evaluate(
        x=test_dataset, y=test_labels, metrics=metrics
    )
    print(eval_results);

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.app.run()