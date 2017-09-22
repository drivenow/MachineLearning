# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:18:52 2016
利用2d卷积层,用x_train训练x_train.
事实上，可以用x_train_noisy加噪声，训练x-train，这样能训练出更鲁棒的特征
@author: Shenjunling
"""
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#读取数据
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
#x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
x_train = np.reshape(x_train, (len(x_train),28, 28,1))
x_test = np.reshape(x_test, (len(x_test), 28, 28,1))


#%%参数

#input_img = Input(shape=(1, 28, 28))
input_shape = Input(shape=(28, 28,1))

#%%
"""
基于CNN的自编码器，三层卷积编码，三层卷积解码
"""
def autoCoderCNN(input_shape):
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_shape)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    
    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
    return decoded

autoCoderCNN_output=autoCoderCNN(input_shape) 
autoencoder = Model(input_shape, autoCoderCNN_output)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
#save model                
#json_string=autoencoder.to_json()
#open("D:/OneDrive/codes/python/nn/model/autoEncoder_mnist.json","w").write(json_string)
#autoencoder.save_weights("D:/OneDrive/codes/python/nn/model/autoEncoder_mnist_weight.h5")

#%%
# encode and decode some digits
decoded_imgs = autoencoder.predict(x_test)
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#对中间层的可视化
#encoder=Model(input=x_test,)