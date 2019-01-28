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
    df = pickle.loads(f)
    
