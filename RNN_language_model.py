"""
-*- coding: utf-8 -*-
Created on Sun Oct 28 22:55:48 2018

@author: shaival
"""

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import numpy as np
from keras.utils import to_categorical

data ='''Jack and Jill went up the hill .
To fetch a pail of water .
Jack fell down and broke his crown .
And Jill came tumbling after .'''

split_data = data.split('\n')

print(split_data)

tokenizer=Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\]^_`{|}~ ',lower=True)
tokenizer.fit_on_texts(split_data)
encoded=tokenizer.texts_to_sequences(split_data)

print(encoded)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

x = list()
y = list()

# make input output pairs for training 
for i in range(0, len(encoded)):
  
  temp_x = list()
  temp_y = list()
  
  for j in range(0,len(encoded[i])-1):
    
    if j==0:
      temp_x.append(0)
      temp_y.append(encoded[i][j])
      
      temp_x.append(encoded[i][j])
      temp_y.append(encoded[i][j+1])
      
      continue   
    
    temp_x.append(encoded[i][j])
    if j!=len(encoded[i])-1:
      temp_y.append(encoded[i][j+1])
      
  x.append(temp_x)
  y.append(temp_y)
  
print(x)
print(y)

temp = list()
y=np.array(y)

from keras.preprocessing.sequence import pad_sequences
max_length = 8
y = pad_sequences(y, maxlen=max_length, padding='post')
x = pad_sequences(x, maxlen=max_length, padding='post')



f=to_categorical(y,vocab_size) # convert to one hot encoded vector
y=f
print(y)
x=np.array(x)

print(y.shape)
print(x)

from keras.layers import SimpleRNN
model = Sequential()
model.add(Embedding(23,10,mask_zero=True))
model.add(SimpleRNN(40,return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=500, verbose=2,batch_size=128)

# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
	in_text, result = seed_text, seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = np.array(encoded)
      # predict a word in the vocabulary
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text, result = out_word, result + ' ' + out_word
	return result

print(generate_seq(model, tokenizer, 'Jack', 6))

pre=model.predict(x[0])
print(x[0])
print(pre)

"""
accuracy = 0.96
"""