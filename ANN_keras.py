# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:12:09 2018

@author: shaival
"""
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.datasets import mnist
from keras.utils import np_utils

(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = trainX.reshape(trainX.shape[0],(trainX.shape[1]*trainX.shape[2]))
testX = testX.reshape(testX.shape[0],(testX.shape[1]*testX.shape[2]))
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255

trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

inp = Input(shape=(trainX.shape[1],))

layer1 = Dense(256, activation = 'relu')(inp)
layer1 = Dropout(0.2)(layer1)
output = Dense(10, activation = 'softmax')(layer1)

model = Model(inp, output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY,
          batch_size = 256,
          epochs = 20,
          validation_data = [testX, testY])

