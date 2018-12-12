
# Importing libraries

import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime


sample_fraction = 1

# changing to dataset location
os.getcwd()
os.chdir('/home/siddharth/Downloads/Dataset/P_projects/Compressor data')


# =============================================================================
# # reading dataset
# start_red = datetime.now()
# print("Dataset reading starts")
# 
# df  = pd.read_excel('30k_records.xlsx') # dataset reading ..
# 
# print("Dataset reading STOPS")
# stop_red = datetime.now()
# 
# print("Dataset reading time "+str(stop_red-start_red))
# 
# # removing unnecessary columns
# df = df.drop(['prop_th_dcy','prop_tq_dcy','hull_dcy','gt_turb_dcy','output_class'],axis = 1)
# 
# 
# #saving dateset in the pickel file
# outfile = open('dataset_pkl_file','wb')   #creates file even if it's not there
# pickle.dump(df,outfile)
# outfile.close()
# =============================================================================

#reading datset from the pickel file
infile = open('dataset_pkl_file','rb')
df = pickle.load(infile)
infile.close()




# creating subsample for testing
df_s = df.sample(frac = sample_fraction,random_state = 101)


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
    train_test_split(scaled_data_X_s, Y_s_ohe, test_size=0.20, random_state=42)
    
# =============================================================================
# # tensorflow code
# =============================================================================
    
import tensorflow as tf

#reset graph

tf.reset_default_graph()

# parameters

num_inputs = 25  
neurons_hid1 = 18 
neurons_hid2 = 9 
neurons_hid3 = 5
num_outputs =15

learning_rate = 0.001

# activation function

act_func = tf.nn.relu


# placeholder

X_ph = tf.placeholder(tf.float32, shape=[None,num_inputs])
Y_ph = tf.placeholder(tf.float32, shape=[None,num_outputs])


# Weights initialization

initializer = tf.variance_scaling_initializer()  # He initializer is used instead of Xavier initialisation


w1 = tf.Variable(initializer([num_inputs, neurons_hid1]), dtype=tf.float32,name='w1')
w2 = tf.Variable(initializer([neurons_hid1, neurons_hid2]), dtype=tf.float32,name='w2')
w3 = tf.Variable(initializer([neurons_hid2, neurons_hid3]), dtype=tf.float32,name='w3')
w4 = tf.Variable(initializer([neurons_hid3, num_outputs]), dtype=tf.float32,name='w4')


# Biases

b1 = tf.Variable(tf.zeros(neurons_hid1),name='b1')
b2 = tf.Variable(tf.zeros(neurons_hid2),name='b2')
b3 = tf.Variable(tf.zeros(neurons_hid3),name='b3')
b4 = tf.Variable(tf.zeros(num_outputs),name='b4')

# layers description

hid_layer1 = act_func(tf.matmul(X_ph, w1) + b1)
hid_layer2 = act_func(tf.matmul(hid_layer1, w2) + b2)
hid_layer3 = act_func(tf.matmul(hid_layer2, w3) + b3)
output_layer = tf.matmul(hid_layer3, w4) + b4


# loss function

loss = tf.reduce_mean(\
                      tf.nn.softmax_cross_entropy_with_logits \
                      ( labels = Y_ph,logits=output_layer) )

l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.005,scale_l2=0.001 )

weights = tf.trainable_variables() # all vars of the graph

weight_not_bias = [ v for v in weights if 'b' not in v.name ]

#regularization_penalty = tf.contrib.layers.apply_regularization(\
#                            l1_l2_regularizer, weight_not_bias)

#l2_regularizer = tf.add_n([ tf.nn.l2_loss(v) for v in weights \
#                           if 'b' not in v.name ]) * 0.001

regularized_loss = loss #+ regularization_penalty 

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

model_start = datetime.now()

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
    
    saver.save(sess, "./model_ckpt1/gas_compressor.ckpt")

model_stop = datetime.now()

print("Model trainging time "+str(model_stop-model_start))

# Evaluating output


with tf.Session() as sess:
    
    saver.restore(sess,"./model_ckpt1/gas_compressor.ckpt")
    
    
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



