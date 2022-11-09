#!/usr/bin/env python

import lightgbm as lgb
import os

# 推論用エンドポイントを作成する際に用いるモデルの読み込み用処理
def model_fn(model_dir):
    print('============== START: model_fn =================')
    print('='*50)
    os.system('pwd')
    print('='*50)
    os.system('ls')
    print('='*50)
    os.system('ls /opt/ml/model')
    print('='*50)

    print('==================================')
    print('======= Display ENV values =======')
    print('==================================')    
    for key in os.environ:
        val = os.environ[key]
        print('{}: {}'.format(key, val))
        
    print('=================================')
    print('=======START LOAD MODEL===========')
    print('=================================')
    model = lgb.Booster(model_file='/opt/ml/model/lightgbm-regression-model.txt')
    
    print('============== END: model_fn =================')
    return model

def predict_fn(transformed_data, model):
    print('============== START: predict_fn =================')
    
    print('====================================')
    print('======= transformed_data ===========')
    print('====================================')
    print(type(transformed_data))
    print(transformed_data)
    
    print('========= predict ============')
    pred = model.predict(transformed_data)
    
    print('===============================================')
    print('============== result of pred =================')
    print('===============================================')
    print(type(pred))
    print(pred)
    
    print('============== END: predict_fn =================')
    return pred

def input_fn(input_data, content_type):
    print('============== START: input_fn =================')
    
    print('====================================')
    print('======= Check input_data ===========')
    print('====================================')
    print(f'content_type={content_type}')
    print(type(input_data))
    print(input_data)
    
    ### リストは二次元配列にする必要がある。まず、'\n'で区切る。その後','で区切る
    if content_type == 'text/csv':
        transformed_data = [s.split(',') for s in input_data.splitlines()]
    else:
        raise ValueError("Illegal content type")

    print('============== END: input_fn =================')
    return transformed_data