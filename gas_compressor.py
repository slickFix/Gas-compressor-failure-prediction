
# Importing libraries

import numpy as np
import pandas as pd
import os

# changing to dataset location
os.getcwd()
os.chdir('/home/siddharth/Downloads/Dataset/P_projects/Compressor data')


# reading dataset
df  = pd.read_excel('30k_records.xlsx')

# removing unnecessary columns
df = df.drop(['prop_th_dcy','prop_tq_dcy','hull_dcy','gt_turb_dcy','output_class'],axis = 1)


# creating subsample for testing
df_s = df.sample(frac = 0.01,random_state = 101)


# creating X and Y variables 

X_s = df_s[['lever', 'speed', 'gt_shaft_tq', 'gt_speed', 'cpp_th', 'cpp_tn',
       'shaft_tq_pt', 'shaft_rpm_pt', 'shaft_tq_Q', 'shaft_rpm_stbd',
       'hp_turb_ex_T', 'gg_speed', 'ff_mf', 'abb_Tic', 'gt_cmpr_outP',
       'gt_cmpr_outT', 'pext_bar', 'hp_turb_outP', 'tcs_signal', 'th_coef_st',
       'prop_rps', 'th_coef_pt', 'prop_rps_pt', 'prop_tq_pt', 'prop_tq_st']]

Y_s = df_s['gt_cmpr_dcy']


# printing X_s columns
X_s.columns

# Label Encoding and Onehot encoding of variables

from sklearn.preprocessing import LabelEncoder
Y_s_le = LabelEncoder().fit_transform(Y_s)

from sklearn.preprocessing import OneHotEncoder
Y_s_ohe = OneHotEncoder().fit_transform(Y_s_le.reshape(-1,1))


#scaling the data

from sklearn.preprocessing import MinMaxScaler
scaled_data_X_s = MinMaxScaler().fit_transform(X_s)


# dividing the sample data into train and test set

from sklearn.model_selection import train_test_split

X_train_s, X_test_s, y_train_s, y_test_s = \
    train_test_split(scaled_data_X_s, Y_s_ohe, test_size=0.80, random_state=42)
    
# =============================================================================
# # tensorflow code
# =============================================================================
    
import tensorflow as tf

#reset graph

tf.reset_default_graph()

# parameters

num_inputs = 25  
neurons_hid1 = 50 
neurons_hid2 = 100 
neurons_hid3 = 50
num_outputs =15

learning_rate = 0.001

# activation function

act_func = tf.nn.relu


# placeholder

X_ph = tf.placeholder(tf.float32, shape=[None,num_inputs])
Y_ph = tf.placeholder(tf.float32, shape=[None,15])


# Weights initialization

initializer = tf.variance_scaling_initializer()  # He initializer is used instead of Xavier initialisation


w1 = tf.Variable(initializer([num_inputs, neurons_hid1]), dtype=tf.float32)
w2 = tf.Variable(initializer([neurons_hid1, neurons_hid2]), dtype=tf.float32)
w3 = tf.Variable(initializer([neurons_hid2, neurons_hid3]), dtype=tf.float32)
w4 = tf.Variable(initializer([neurons_hid3, num_outputs]), dtype=tf.float32)


# Biases

b1 = tf.Variable(tf.zeros(neurons_hid1))
b2 = tf.Variable(tf.zeros(neurons_hid2))
b3 = tf.Variable(tf.zeros(neurons_hid3))
b4 = tf.Variable(tf.zeros(num_outputs))

# layers description

hid_layer1 = act_func(tf.matmul(X_ph, w1) + b1)
hid_layer2 = act_func(tf.matmul(hid_layer1, w2) + b2)
hid_layer3 = act_func(tf.matmul(hid_layer2, w3) + b3)
output_layer = tf.matmul(hid_layer3, w4) + b4


# loss function

loss = tf.reduce_mean(\
                      tf.nn.softmax_cross_entropy_with_logits \
                      ( labels = Y_ph,logits=output_layer) )

l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005)
weights = tf.trainable_variables() # all vars of the graph
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
regularized_loss = loss + regularization_penalty

# optimizer

optimizer = tf.train.AdamOptimizer(learning_rate)

# defining the train variable

train = optimizer.minimize(regularized_loss)

# initialising the variables

init = tf.global_variables_initializer()

saver = tf.train.Saver() 

# Training tensoflow model

num_epochs = 5
batch_size = 10

with tf.Session() as sess:
    
    sess.run(init)
    
    for epoch in range(num_epochs):
        
        no_batches = len(X_train_s) // batch_size
        
        for step in range(no_batches):
                    
            rand_ind = np.random.randint(len(X_train_s),size=batch_size)

            feed = {X_ph:X_train_s[rand_ind].astype(np.float32), 
                    Y_ph:y_train_s[rand_ind].toarray().astype(np.float32) }   

            sess.run(train,feed_dict = feed)
            
        training_loss = regularized_loss.eval(feed_dict=feed)   
        
        print("Epoch {} Complete. Training Loss: {}".format(epoch,training_loss))
    
    saver.save(sess, "./model_ckpt/gas_compressor.ckpt")

# Evaluating output


with tf.Session() as sess:
    
    saver.restore(sess,"./model_ckpt/gas_compressor.ckpt")
    
    
    feed = {X_ph:X_test_s.astype(np.float32),  
            Y_ph:y_test_s.toarray().astype(np.float32) }
    
    results = output_layer.eval(feed_dict=feed)
    
    
# Getting 1 D array of y_test

y_test_class_s = [i.argmax() for i in y_test_s]


# Getting 1 D array of y_pred

y_test_result_class_s = [i.argmax() for i in results]


# Checking the accuracy score
from sklearn.metrics import accuracy_score

accuracy_score(y_test_class_s,y_test_result_class_s)



