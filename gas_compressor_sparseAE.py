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

from old_utils import *

from datetime import datetime

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
    
    with tf.variable_scope('Placeholders'):
        x_ph = tf.placeholder(tf.float32,[None,n_x],name='X_ph')    
        y_ph = tf.placeholder(tf.float32,[None,n_y],name='Y_ph')
    
    return x_ph,y_ph


def initialise_parameter_ae_1_2(n_x,n_y):
    ''' layer 1 and layer 2 of the FC is trained by ae '''
    # defining parameters
    input_feat = n_x
    hidden1_nodes = 50
    hidden2_nodes = 100
    hidden3_nodes = 50
    output_feat = n_y
    
    # defining variable initializer
    var_init = tf.variance_scaling_initializer()
    
    with tf.variable_scope('weights'):
        # defining weights 
        W1 = tf.Variable(var_init([input_feat,hidden1_nodes]),dtype=tf.float32,name='w1')
        W2 = tf.Variable(var_init([hidden1_nodes,hidden2_nodes]),dtype=tf.float32,name='w2')
        W3 = tf.Variable(var_init([hidden2_nodes,hidden3_nodes]),dtype=tf.float32,name='w3')
        W4 = tf.Variable(var_init([hidden3_nodes,output_feat]),dtype=tf.float32,name='w4')
    
        # defining tied weights
        W_AE_1 = tf.transpose(W2,name='w_ae_1')
        W_AE = tf.transpose(W1,name='w_ae')
        
        
    with tf.variable_scope('bias'):
        # defining bias
        b1 = tf.Variable(tf.zeros(hidden1_nodes),name='b1')
        b2 = tf.Variable(tf.zeros(hidden2_nodes),name='b2')
        b3 = tf.Variable(tf.zeros(hidden3_nodes),name='b3')
        b4 = tf.Variable(tf.zeros(output_feat),name='b4')
        
        B_AE_1 = tf.Variable(tf.zeros(hidden1_nodes),name='b_ae_1')
        B_AE = tf.Variable(tf.zeros(input_feat),name='b_ae')

    parameters = {            
            'W1':W1,
            'W2':W2,
            'W3':W3,
            'W4':W4,
            'W_AE_1':W_AE_1,
            'W_AE':W_AE,
            'b1':b1,
            'b2':b2,
            'b3':b3,
            'b4':b4,
            'B_AE_1':B_AE_1,
            'B_AE':B_AE}
    return parameters


def fwd_propagation_ae_1_2(x_ph,parameters):
    ''' layer 1 and layer 2 of the FC is trained by ae '''
    
    
    W1 = parameters['W1']
    W2 = parameters['W2'] 
    W3 = parameters['W3'] 
    W4 = parameters['W4']
    W_AE_1 = parameters['W_AE_1'] 
    W_AE = parameters['W_AE'] 
    
    
    b1 = parameters['b1']  
    b2 = parameters['b2']  
    b3 = parameters['b3']  
    b4 = parameters['b4'] 
    B_AE_1 = parameters['B_AE_1'] 
    B_AE = parameters['B_AE']  
    
    act_fn = tf.nn.relu
    
    with tf.variable_scope('layer_1'):
        hid_layer1 = act_fn(tf.add(tf.matmul(x_ph,W1),b1))
    with tf.variable_scope('layer_2'):
        hid_layer2 = act_fn(tf.add(tf.matmul(hid_layer1,W2),b2))
    with tf.variable_scope('layer_3'):
        hid_layer3 = act_fn(tf.add(tf.matmul(hid_layer2,W3),b3))
        
    with tf.variable_scope('output_layer'):
        output_layer = tf.add(tf.matmul(hid_layer3,W4),b4)
    

    with tf.variable_scope('decoder_layer_1'):
        decoder_layer_1 = tf.add(tf.matmul(hid_layer2,W_AE_1),B_AE_1)
        
    with tf.variable_scope('x_hat_layer'):
        x_hat = tf.nn.sigmoid(tf.add(tf.matmul(decoder_layer_1,W_AE),B_AE))
    
    return output_layer,x_hat,hid_layer1,hid_layer2


def compute_cost_fc(logits,y_ph,reg_lambda):
    
    with tf.variable_scope('FC_loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_ph)
        
        weights = tf.trainable_variables()
        weights_without_bias = [v for v in weights if 'b' not in v.name and 'ae' not in v.name] 
        
        l2_loss = reg_lambda*tf.add_n([tf.nn.l2_loss(v) for v in weights_without_bias])
        
        ret_loss = tf.reduce_mean(loss+l2_loss)
    
    return ret_loss

def kl_divergence(p,p_hat):
    return p*tf.log(p)-p*tf.log(p_hat)+(1-p)*tf.log(1-p)-(1-p)*tf.log(1-p_hat)

def compute_cost_ae_1_2(x_hat,x_ph,parameters,reg_lambda,hid_layer1,hid_layer2,rho,beta):
    
    with tf.variable_scope('AE_loss'):
        diff = x_hat-x_ph
        
        p_hat1 = tf.reduce_mean(tf.clip_by_value(hid_layer1,1e-10,1.0,name='clipper'),axis = 0)
        p_hat2 = tf.reduce_mean(tf.clip_by_value(hid_layer2,1e-10,1.0,name='clipper'),axis = 0)    
        
        kl1 = kl_divergence(rho,p_hat1)
        kl2 = kl_divergence(rho,p_hat2)
        
        kl_loss = beta*(tf.reduce_sum(kl1)+tf.reduce_sum(kl2))
        
        W1 = parameters['W1']
        W2 = parameters['W2'] 
        W3 = parameters['W3']
        W_AE_1 = parameters['W_AE_1']
        W_AE = parameters['W_AE']
        l2_loss = reg_lambda*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(W_AE_1)+tf.nn.l2_loss(W_AE))
        
        loss = tf.reduce_mean(tf.reduce_sum(diff**2,axis=1))+kl_loss+l2_loss
    
    return loss


def compute_cost_ae(x_hat,x_ph,parameters,reg_lambda,hid_layer1,rho,beta):
    
    diff = x_hat-x_ph
    
    p_hat = tf.reduce_mean(tf.clip_by_value(hid_layer1,1e-10,1.0,name='clipper'),axis = 0)    
    kl = kl_divergence(rho,p_hat)
    
    W1 = parameters['W1']
    W_AE = parameters['W_AE']
    l2_loss = reg_lambda*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W_AE))
    
    loss = tf.reduce_mean(tf.reduce_sum(diff**2,axis=1))+beta*tf.reduce_sum(kl)+l2_loss
    
    return loss
       
def model(X_train_scaled_s,y_train_s,X_test_s,y_test_s,sc,learning_rate = 1e-3,rho=0.1,beta=3,reg_lambda = 1e-6,n_epochs = 50,batch_size=100):
    ''' 
    Function for executing the NN model
    
    Args:-
    
    arg1 : x_train \n
    arg2 : y_train \n
    arg3 : x_test \n
    arg4 : y_test \n
    arg5 : sc \n
    arg6 : learning rate(default) \n
    arg7 : rho(default) \n
    arg8 : beta(default) \n
    arg9 : reg_lambda(default) \n
    arg10 : n_epochs(default) \n
    arg11: batch_size(default) \n
    '''
    
    # reset default graph
    tf.reset_default_graph()
    
    # getting the x and y features
    n_x = X_train_scaled_s.shape[1]
    n_y = y_train_s.shape[1]
    
    # creating placeholders
    x_ph,y_ph = create_placeholder(n_x,n_y)
    
    # parameters initialisation
    #parameters = initialise_parameter(n_x,n_y)
    #parameters = initialise_parameter_ae_all(n_x,n_y)
    parameters = initialise_parameter_ae_1_2(n_x,n_y)
    
    # forward propagation
    #logits,x_hat,hid_layer1 = fwd_propagation(x_ph,parameters)
    #logits,x_hat,hid_layer1,hid_layer2,hid_layer3 = fwd_propagation_ae_all(x_ph,parameters)
    logits,x_hat,hid_layer1,hid_layer2 = fwd_propagation_ae_1_2(x_ph,parameters)
    
    # cost history saving
    cost_fc_li = []
    cost_ae_li = []
    
    # cost calculation
    cost_fc = compute_cost_fc(logits,y_ph,reg_lambda)
    #cost_ae = compute_cost_ae(x_hat,x_ph,parameters,reg_lambda,hid_layer1,rho,beta)
    #cost_ae = compute_cost_ae_all(x_hat,x_ph,parameters,reg_lambda,hid_layer1,hid_layer2,hid_layer3,rho,beta)
    cost_ae = compute_cost_ae_1_2(x_hat,x_ph,parameters,reg_lambda,hid_layer1,hid_layer2,rho,beta)
    
    # optimizers
    optimizer_fc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_fc)
    optimizer_ae = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_ae)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        
        print('Training the Sparse Auto encoder')
        for epoch in range(n_epochs):            
            n_batches = len(X_train_scaled_s)//batch_size
            epoch_cost = 0
            for step in range(n_batches+1):
                feed = {}
                if step != n_batches:
                    feed[x_ph] = X_train_scaled_s[step*batch_size:(step+1)*batch_size]
                else:
                    feed[x_ph] = X_train_scaled_s[step*batch_size:]
               
                _,c_ae = sess.run([optimizer_ae,cost_ae],feed)
                epoch_cost += c_ae/n_batches
            
            cost_ae_li.append(epoch_cost)
            if (epoch+1)%2 == 0:
                print("Epoch: ",epoch+1," cost: ",epoch_cost)
            
            
        print('Training FC model with AE feature in the 1st layer and 2nd layer')
        for epoch in range(n_epochs):
            n_batches = len(X_train_scaled_s) // batch_size
            epoch_cost = 0 
            for step in range(n_batches +1):
                feed = {}
                if step != n_batches:
                    feed[x_ph] = X_train_scaled_s[step*batch_size:(step+1)*batch_size]
                    feed[y_ph] = y_train_s[step*batch_size:(step+1)*batch_size]
                else:
                    feed[x_ph] = X_train_scaled_s[step*batch_size:]
                    feed[y_ph] = y_train_s[step*batch_size:]
                
                _, c_fc = sess.run([optimizer_fc,cost_fc],feed)
                epoch_cost += c_fc/n_batches
            
            cost_fc_li.append(epoch_cost)
            
            if (epoch+1)%2 == 0:
                print("Epoch: ",epoch+1," cost: ",epoch_cost)
                
# =============================================================================
#             if epoch == 0:
#                 past_training_loss = epoch_cost
#                 continue
#             
#             elif epoch_cost < past_training_loss:# if new loss is less than past loss "save new model parameters"            
#                 saver.save(sess, "./model_ae_fc_1_2/ae_99.8_fc",global_step=epoch+1)
#                 print(f'saving model for epoch : {epoch+1}')
#                 past_training_loss = epoch_cost
# =============================================================================
            
        with tf.variable_scope('accuracy_cal'):
            correct_pred = tf.equal(tf.math.argmax(logits,axis=1),tf.math.argmax(y_ph,axis=1))
            acc = tf.reduce_mean(tf.cast(correct_pred,'float'))
        
        feed_train ={}
        feed_train[x_ph] = X_train_scaled_s.astype(np.float32)
        feed_train[y_ph] = y_train_s.astype(np.float32)
        
        print("Training accuracy of the AE_FC model is : ",acc.eval(feed_train))
        
        feed_test = {}
        X_test_scaled_s = sc.transform(X_test_s)  # scaling the test set
        feed_test[x_ph] = X_test_scaled_s.astype(np.float32)
        feed_test[y_ph] = y_test_s.astype(np.float32)
        
        print("Test accuracy of the AE_FC model is : ", acc.eval(feed_test))
        
# =============================================================================
#         writer = tf.summary.FileWriter('./tensorboard_ae_fc_1_2',sess.graph)
#         writer.close()
# =============================================================================
        

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
    
    # TRAINING TF MODEL
    training_start = datetime.now()
    
    #model(X_train_scaled_s,y_train_s,X_test_s,y_test_s,sc,learning_rate = 0.001,n_epochs = 50,batch_size=100)
    
    training_stop = datetime.now()
    print('Training time for the AE+FC model: ',str(training_stop-training_start))
    
    
    # model evaluation on 40 % not seen dataset
    
    x_eval = df_not_s.iloc[:,:-1]
    y_eval = df_not_s.iloc[:,-1]
    
    # label encoding the output variable
    le_eval = LabelEncoder()
    y_eval = le_eval.fit_transform(y_eval)
    
    # one-hot encoding of the dependent variable
    y_eval = OneHotEncoder().fit_transform(y_eval.reshape(-1,1)) # it's a scipy.sparse.csr.csr_mat
    y_eval = y_eval.toarray() # converting to ndarray

    # accuracy on eval dataset    
    with tf.Session() as sess:
        
        # as 46th step gives the best result
        saver = tf.train.import_meta_graph('./model_ae_fc_1_2/ae_99.8_fc-46.meta')
        saver.restore(sess,'./model_ae_fc_1_2/ae_99.8_fc-46')
        
        graph = tf.get_default_graph()
        
        y_ph = graph.get_tensor_by_name('Placeholders/Y_ph:0')
        x_ph = graph.get_tensor_by_name('Placeholders/X_ph:0')
        logits = graph.get_tensor_by_name('output_layer/Add:0')
        
        correct_pred = tf.equal(tf.math.argmax(logits,axis=1),tf.math.argmax(y_ph,axis=1))
        acc = tf.reduce_mean(tf.cast(correct_pred,'float'))
        
        feed_eval = {}
        X_eval_scaled = sc.transform(x_eval)  #scaling the eval set
        feed_eval[x_ph] = X_eval_scaled.astype(np.float32)
        feed_eval[y_ph] = y_eval.astype(np.float32)
        
        print("Accuracy for the 40% unseen data is : ", acc.eval(feed_eval))



