#!/usr/bin/env python

#from __future__ import print_function
import lightgbm as lgb
#import argparse
import os
#from os import path
#import logging
#import pickle
#import json
#import pandas as pd
#import glob

# 推論用エンドポイントを作成する際に用いるモデルの読み込み用処理
def model_fn(model_dir):
    print('=================================')
    print('=================================')
    print('==========my model_fn==============')
    print('=================================')
    print('=================================')
    
    os.system('pwd')
    os.system('ls')
    print('==============a===================')
    os.system('ls /opt/ml/model')
    print('===============a==================')
    
    
    print('==================================')
    print('======= Display ENV values =======')
    print('==================================')    

    for key in os.environ:
        val = os.environ[key]
        print('{}: {}'.format(key, val))
    print('=================================')
    print('=======START LOAD MODEL===========')
    print('=================================')
    #with open(path.join(model_dir, 'model.pickle'), 'rb') as f:
    #    model = pickle.load(f)
    model = lgb.Booster(model_file='/opt/ml/model/lightgbm-regression-model.txt')
    print('=================================')
    print('=======END OF model_fn===========')
    print('=================================')
    return model

def predict_fn(transformed_data, model):
    print('=================================')
    print('=================================')
    print('==========my predict_fn =========')
    print('=================================')
    print('=================================')
    
    print('=================================')
    print('=======transformed_data===========')
    print('=================================')
    print(type(transformed_data))
    print(transformed_data)
    print('=================================')
    
    pred = model.predict(transformed_data)
    print('===============================================')
    print('============== result of pred =================')
    print('===============================================')
    print(type(pred))
    print(pred)
    
    print('============== END: result of pred =================')
    
    return pred

def input_fn(input_data, content_type):
    print('=================================')
    print('=================================')
    print('==========my input_fn =========')
    print('=================================')
    print('=================================')
    ### リストは二次元配列にする必要がある。
    ### まず、\nで区切る
    ### , で区切る
    print(content_type)
    print('========== input_data =========')
    print(type(input_data))
    print(input_data)
    print('========== END: input_data =========')
    if content_type == 'text/csv':
        transformed_data = input_data.splitlines()
        #transformed_data = input_data.split(',')
        print(type(transformed_data))
        print(transformed_data)
        print(len(transformed_data)) # 102行
        print('========== END: transformed_data =========')
        transformed_data = [s.split(',') for s in input_data.splitlines()]
        
    else:
        raise ValueError("Illegal content type")
    return transformed_data