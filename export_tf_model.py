#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:13:47 2019

@author: siddharth
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf

# defining paths
dataset_path = '/home/siddharth/Downloads/Dataset/P_projects/Compressor data'
os.chdir(dataset_path)

saved_model_dir = './model_ae_fc_1_2/ae_99.8_fc-46'
export_dir = os.path.join('export_model','0')

# tf code
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    # restoring the checkpoint
    saver = tf.train.import_meta_graph(saved_model_dir+'.meta')
    saver.restore(sess,saved_model_dir)
    
    # creating the model input and output information
    output_tensor = sess.graph.get_tensor_by_name('output_layer/Add:0')
    input_tensor = sess.graph.get_tensor_by_name('Placeholders/X_ph:0')
    
    model_input = tf.saved_model.utils.build_tensor_info(input_tensor)
    model_output = tf.saved_model.utils.build_tensor_info(output_tensor)
    
    # creating the model signature definition
    prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'in':model_input},
            outputs={'out':model_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    
    # export checkpoint to the saved model
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                                 tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature},
                                                 strip_default_attrs=True)
    builder.save()