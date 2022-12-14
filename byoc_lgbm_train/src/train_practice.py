#!/usr/bin/env python

# A sample training component that trains a simple LightGBM Regression model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

import logging
import os
import json
import traceback
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

prefix = '/opt/ml/'
input_path = prefix + 'input/data'
train_channel_name = 'train'
validation_channel_name = 'validation'

output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
model_file_name = 'lightgbm-regression-model.txt'
train_path = os.path.join(input_path, train_channel_name)
validation_path = os.path.join(input_path, validation_channel_name)

param_path = os.path.join(prefix, 'input/config/hyperparameters.json')


# The function to execute the training.
def train():
    print('===== congraturations! You understand how SageMaker Training Job Works!!! =====')
    
    print('Starting the training.')

    try:
        # Read in any hyperparameters that the user passed with the training job
        print('Reading hyperparameters data: {}'.format(param_path))
        with open(param_path) as json_file:
            hyperparameters_data = json.load(json_file)
        print('hyperparameters_data: {}'.format(hyperparameters_data))

        # Take the set of train files and read them all into a single pandas dataframe
        train_input_files = [os.path.join(train_path, file) for file in os.listdir(train_path)]
        if len(train_input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(train_path, train_channel_name))
        print('Found train files: {}'.format(train_input_files))
        raw_data = [pd.read_csv(file) for file in train_input_files]
        train_df = pd.concat(raw_data)

        # Take the set of train files and read them all into a single pandas dataframe
        validation_input_files = [os.path.join(validation_path, file) for file in os.listdir(validation_path)]
        if len(validation_input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(validation_path, train_channel_name))
        print('Found validation files: {}'.format(validation_input_files))
        raw_data = [pd.read_csv(file) for file in validation_input_files]
        validation_df = pd.concat(raw_data)

        # Assumption is that the label is the last column
        print('building training and validation datasets')
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1:]
        X_validation = validation_df.iloc[:, :-1]
        y_validation = validation_df.iloc[:, -1:]

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            'boosting_type': hyperparameters_data['boosting_type'],
            'objective': hyperparameters_data['objective'],
            'num_leaves': hyperparameters_data['num_leaves'],
            'learning_rate': hyperparameters_data['learning_rate'],
            'feature_fraction': hyperparameters_data['feature_fraction'],
            'bagging_fraction': hyperparameters_data['bagging_fraction'],
            'bagging_freq': hyperparameters_data['bagging_freq'],
            'verbose': hyperparameters_data['verbose']
        }

        print('Starting training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=20,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=5)

        # persist model
        path = os.path.join(model_path, model_file_name)
        print('saving model file to {}'.format(path))
        # save model to file
        gbm.save_model(path)

        print('Training complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc)
        # A non-zero exit dependencies causes the training job to be marked as Failed.
        sys.exit(255)
    

if __name__ == '__main__':
    train()

    # A zero exit dependencies causes the job to be marked a Succeeded.
    sys.exit(0)
