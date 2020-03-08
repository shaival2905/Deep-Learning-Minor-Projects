# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 12:09:13 2018

@author: shaival
"""

import numpy as np


X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

y=np.array([[1,0],[1,1],[0,1]])

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

epoch=6000
lr=0.1
inputlayer = X.shape[1]
hiddenlayer = 5 
output = 2

w=np.random.uniform(size=(inputlayer,hiddenlayer))
b=np.random.uniform(size=(1,hiddenlayer))
w_out=np.random.uniform(size=(hiddenlayer,output))
b_out=np.random.uniform(size=(1,output))

for i in range(epoch):

    outh=np.dot(X,w) + b 
    outh = sigmoid(outh)
    output = np.dot(outh,w_out) + b_out
    output = sigmoid(output)
    
    E = y-output
    d_output = E * derivatives_sigmoid(output)
    d_hidden = derivatives_sigmoid(outh)
    err_hidden = d_output.dot(w_out.T)
    d_hiddenlayer = err_hidden * d_hidden
    w_out += outh.T.dot(d_output) *lr
    b_out += np.sum(d_output, axis=0,keepdims=True) *lr
    w += X.T.dot(d_hiddenlayer) *lr
    b += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print (output)