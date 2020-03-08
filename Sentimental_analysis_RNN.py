# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:55:48 2018

@author: shaival
"""

from keras.datasets import imdb
vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size) #get training and testing data fromm imdb keras dataset

word2id = imdb.get_word_index() #dictionary of words corresponding to number assigned
id2word = {i: word for word, i in word2id.items()}# reverse dictionary of word2id for prediction

print('Maximum review length: {}'.format(
len(max((X_train + X_test), key=len))))

print('Minimum review length: {}'.format(
len(min((X_test + X_test), key=len))))

from keras.preprocessing import sequence
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)# pad sequence to make all of same length
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
embedding_size=32 #embedding dimention
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
print(model.summary()) #get summary of whole model

model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

batch_size = 64
num_epochs = 3
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size] #split into training and testing
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

scores = model.evaluate(X_test, y_test, verbose=0) #get model score loss and accuracy of model
print('Test accuracy:', scores[1])
print(model.predict_classes(X_test))# predict the class for testing data

"""
validation accuracy: 0.80
"""