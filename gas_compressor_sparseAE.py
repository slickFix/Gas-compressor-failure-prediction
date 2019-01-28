#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:18:05 2019

@author: siddharth
"""

import numpy as np
import pandas as pd

import tensorflow as tf
import pickle
import os


# Changing CWD to the dataset (pickle) location
os.getcwd()
os.chdir('/home/siddharth/Downloads/Dataset/P_projects/Compressor data')


with open('dataset_pkl_file','rb') as f:
    df = pickle.load(f)
    

# taking the sample of the dataset
sample_fraction = 0.6

df_s = df.sample(frac = sample_fraction,random_state = 101)

df_not_s = df.iloc[[i for i in df.index if i not in df_s.index]]

# creating the dependent and independent variables
x_s = df_s.iloc[:,:-1]
y_s = df_s.iloc[:,-1]

# label encoding the output variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_s = le.fit_transform(y_s)


# splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(x_s, y_s, test_size=0.20, random_state=42)

# Scaling the input data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled_s = sc.fit_transform(X_train_s)

# =============================================================================
# tensorflow code
# =============================================================================


# reset tensorflow graph
tf.reset_default_graph()


# defining parameters
input_feat = 25
hidden1_nodes = 50
hidden2_nodes = 100
hidden3_nodes = 50
output_feat = 1

learning_rate = 0.001

# defining activation function
act_func = tf.nn.relu

# defining placeholder
x_ph = tf.placeholder(tf.float32,[None,input_feat],'x_ph')
y_ph = tf.placeholder(tf.float32,[None,output_feat],'y_ph')

# defining variable initializer
var_init = tf.variance_scaling_initializer()

# defining weights and bias
W1 = tf.Variable(var_init([input_feat,hidden1_nodes]),dtype=tf.float32,name='w1')
W2 = tf.Variable(var_init([hidden1_nodes,hidden2_nodes]),dtype=tf.float32,name='w2')
W3 = tf.Variable(var_init([hidden2_nodes,hidden3_nodes]),dtype=tf.float32,name='w3')
W4 = tf.Variable(var_init([hidden3_nodes,output_feat]),dtype=tf.float32,name='w4')

W_AE = tf.Variable(var_init([hidden1_nodes,input_feat]),dtype=tf.float32,name='w_ae')

b1 = tf.Variable(tf.zeros(hidden1_nodes),name='b1')
b2 = tf.Variable(tf.zeros(hidden1_nodes),name='b2')
b3 = tf.Variable(tf.zeros(hidden1_nodes),name='b3')
b4 = tf.Variable(tf.zeros(hidden1_nodes),name='b4')

B_AE = tf.Variable(tf.zeros(input_feat),name='b_ae')




