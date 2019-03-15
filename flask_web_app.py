#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:18:39 2019

@author: siddharth
"""

import os
import sys
import logging
import json

from flask import Flask,request,jsonify
from flask_cors import CORS,cross_origin

from translate_client import Server

HOST = '0.0.0.0'

# Defining the app
app = Flask(__name__)
CORS(app)

# Loading the model 
server = Server(HOST,9000)

# API ROUTE

@app.route('/api',methods=['POST'])
@cross_origin()
def api():
    
    input_data = request.json
    app.logger.info('api input: '+str(input_data))
    
    payload = input_data
    
    output_data = server.translate(payload['data'])
    
    app.logger.info('api output: '+ str(output_data))
    
    print('output data type: '+str(type(output_data)))
    
    response = jsonify(output_data.tolist())
    
    return response

@app.route('/')
def index():
    return 'Index API'


@app.errorhandler(404)
def url_error(e):
    return "Wrong URL <pre>{}</pre>".format(e)

@app.errorhandler(500)
def server_error(e):
    return "An internal server error has occured <pre>{}</pre>".format(e)


if __name__ == '__main__':
    app.run(host=HOST,debug = True)
    
    
    
    