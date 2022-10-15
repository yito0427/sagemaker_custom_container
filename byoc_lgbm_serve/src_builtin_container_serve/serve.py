#!/usr/bin/env python

from __future__ import print_function
import lightgbm as lgb
import argparse
import os
from os import path
import logging
import pickle
import json
import pandas as pd
import glob

# 推論用エンドポイントを作成する際に用いるモデルの読み込み用処理
def model_fn(model_dir):
    with open(path.join(model_dir, 'model.pickle'), 'rb') as f:
        model = pickle.load(f)
    return model
