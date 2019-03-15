#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:35:53 2019

@author: siddharth
"""

import pandas as pd
import numpy as np

import os 
import json
import requests
import pickle

path = '/home/siddharth/Downloads/Dataset/P_projects/Compressor data'
os.chdir(path)

filename = 'dataset_pkl_file'

if __name__ == '__main__':
    
    print('Data loading'.center(50,'-'))
    
    with open(filename,'rb') as f:
        df = pickle.load(f)
        
    print('Preprocessing starts'.center(50,'-'))
    
    x = df.iloc[:10,:-1] # taking small subset of the data
    
    with open('sc.pkl','rb') as f:
        sc = pickle.load(f)
    
    x_scale = sc.transform(x)
    
    # converting x_scale to list for JSONIFICATION
    x_scale = x_scale.tolist()
    
    print("Making the inference call".center(50,'-'))
    
    url = 'http://0.0.0.0:5000/api'
    payload = {'data':x_scale}
    headers = {'content-type':'application/json'}
    
    r = requests.post(url = url,data = json.dumps(payload),headers=headers)
    
    print("response code: ",r.status_code)
    print('Content received from prediction ',r.text)
    print('End of program'.center(50,'-'))   
    
    
    