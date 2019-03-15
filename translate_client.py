#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:45:09 2019

@author: siddharth
"""

import numpy as np

# Communication to Tensorflow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

# Tensorflow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto


timeout = 60.0

class Server:
    def __init__(self,host,port):
        # channel and stub are boiler-plate:
        channel = implementations.insecure_channel(host,int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        
    def translate(self,record):
        
        print('record length: ',len(record))        
        m = len(record)
        
        # boiler plate
        request = predict_pb2.PredictRequest()
        
        # set request objects using the tf-serving 'CopyFrom' setter methods
        request.model_spec.name = '0'
        request.model_spec.signature_name = 'serving_default'
        request.inputs['in'].CopyFrom(make_tensor_proto(record,shape=[m,len(record[0])]))
        
        # boiler plate
        response= self.stub.Predict(request,timeout)
        
        result = response.outputs['out']
        result_arr = tf.make_ndarray(result)
        
        
        return result_arr
                
        
        