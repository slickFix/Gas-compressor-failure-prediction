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


