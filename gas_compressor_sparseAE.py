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


def get_data():
    
    with open('dataset_pkl_file','rb') as f:
        df = pickle.load(f)
         
    # taking the sample of the dataset
    sample_fraction = 0.6
    
    df_s = df.sample(frac = sample_fraction,random_state = 101)
    
    df_not_s = df.iloc[[i for i in df.index if i not in df_s.index]]
    
    # creating the dependent and independent variables
    x_s = df_s.iloc[:,:-1]
    y_s = df_s.iloc[:,-1]
    
    return x_s,y_s,df_not_s


def create_placeholder(n_x,n_y):
    
    x_ph = tf.placeholder(tf.float32,[None,n_x],name='X_ph')    
    y_ph = tf.placeholder(tf.float32,[None,n_y],name='Y_ph')
    
    return x_ph,y_ph

def initialise_parameter(n_x,n_y):
    
    # defining parameters
    input_feat = n_x
    hidden1_nodes = 50
    hidden2_nodes = 100
    hidden3_nodes = 50
    output_feat = n_y
    
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

    parameters = {            
            'W1':W1,
            'W2':W2,
            'W3':W3,
            'W4':W4,
            'W_AE':W_AE,
            'b1':b1,
            'b2':b2,
            'b3':b3,
            'b4':b4,
            'B_AE':B_AE}
    return parameters

def fwd_propagation(x_ph,parameters):
    
    W1 = parameters['W1']
    W2 = parameters['W2'] 
    W3 = parameters['W3'] 
    W4 = parameters['W4']  
    W_AE = parameters['W_AE'] 
    b1 = parameters['b1']  
    b2 = parameters['b2']  
    b3 = parameters['b3']  
    b4 = parameters['b4'] 
    B_AE = parameters['B_AE'] 
    
    act_fn = tf.nn.relu
    
    hid_layer1 = act_fn(tf.add(tf.matmul(x_ph,W1),b1))
    hid_layer2 = act_fn(tf.add(tf.matmul(hid_layer1,W2),b2))
    hid_layer3 = act_fn(tf.add(tf.matmul(hid_layer2,W3),b3))
    
    output_layer = tf.add(tf.matmul(hid_layer3,W4),b4)
    
    x_hat = tf.nn.sigmoid(tf.add(tf.matmul(hid_layer1,W_AE),B_AE))
    
    return output_layer,x_hat,hid_layer1
    
def compute_cost_fc(logits,y_ph,reg_lambda):
    
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_ph)
    
    weights = tf.trainable_variables()
    weights_without_bias = [v for v in weights if 'b' not in v.name and 'ae' not in v.name] 
    
    l2_loss = reg_lambda*tf.add_n([tf.nn.l2_loss(v) for v in weights_without_bias])
    
    return tf.reduce_mean(loss+l2_loss)
    
    
def model(X_train_scaled_s,y_train_s,X_test_s,y_test_s,learning_rate = 1e-3,reg_lambda = 1e-6,n_epochs = 50):
    
    # reset default graph
    tf.reset_default_graph()
    
    # getting the x and y features
    n_x = X_train_scaled_s.shape[1]
    n_y = y_train_s.shape[1]
    
    
    # creating placeholders
    x_ph,y_ph = create_placeholder(n_x,n_y)
    
    # parameters initialisation
    parameters = initialise_parameter(n_x,n_y)
    
    # forward propagation
    logits,x_hat,hid_layer1 = fwd_propagation(x_ph,parameters)
    
    # cost calculation
    cost_fc = compute_cost_fc(logits,y_ph,reg_lambda)
    cost_ae = compute_cost_ae(x_hat,x_ph)
    

if __name__ == '__main__':
    
    # getting the data as well as "DF_NOT_S"
    x_s,y_s,df_not_s = get_data()
    
    # label encoding the output variable
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_s = le.fit_transform(y_s)
    
    # one-hot encoding of the dependent variable
    from sklearn.preprocessing import OneHotEncoder
    y_s = OneHotEncoder().fit_transform(y_s.reshape(-1,1)) # it's a scipy.sparse.csr.csr_mat
    y_s = y_s.toarray() # converting to ndarray
    
    # splitting the data into train and test set
    from sklearn.model_selection import train_test_split
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(x_s, y_s, test_size=0.20, random_state=42)
    
    # Scaling the input data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_train_s)
    X_train_scaled_s = sc.transform(X_train_s)
    
    
    model(X_train_scaled_s,y_train_s,X_test_s,y_test_s,learning_rate = 0.001,n_epochs = 50)

    





