# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:13:43 2018

@author: shaival
"""

from keras.models import Sequential
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

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(trainX.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY,
          batch_size = 512,
          epochs = 20,
          validation_data = [testX, testY])

score = model.evaluate(testX, testY)