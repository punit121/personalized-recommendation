#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sagemaker
from pathlib import Path
from sagemaker.predictor import json_serializer
#from container.bert.constants.constant import *
import json


def train_deploy_personalized_recommendation():

    role = sagemaker.get_execution_role()
    session = sagemaker.Session()

    # location for train.csv, val.csv and labels.csv
    DATA_PATH = Path("../data/")

    # Location for storing training_config.json
    CONFIG_PATH = DATA_PATH/'config'
    CONFIG_PATH.mkdir(exist_ok=True)

    # S3 bucket name
    bucket = 'personalized-recommendation'

    # Prefix for S3 bucket for input and output
    prefix = 'personalized-recommendation/input'
    prefix_output = 'personalized-recommendation/output'


   

    # This is a  feature to upload data to S3 bucket

    s3_input = session.upload_data(DATA_PATH, bucket=bucket , key_prefix=prefix)

    session.upload_data(str(DATA_PATH/'train.csv'), bucket=bucket , key_prefix=prefix)
    session.upload_data(str(DATA_PATH/'val.csv'), bucket=bucket , key_prefix=prefix)

    #  Creating an Estimator and start training

    account = session.boto_session.client('sts').get_caller_identity()['Account']
    region = session.boto_session.region_name

    image = "{}.dkr.ecr.{}.amazonaws.com/fluent-sagemaker-fast-bert:1.0-gpu-py36".format(account, region)

    output_path = "s3://{}/{}".format(bucket, prefix_output)

    estimator = sagemaker.estimator.Estimator(image,
                                              role,
                                              train_instance_count=1,
                                              train_instance_type='ml.p2.xlarge',
                                              output_path=output_path,
                                              base_job_name='bert-text-classification-v1',
                                              hyperparameters=hyperparameters,
                                              sagemaker_session=session
                                             )

    estimator.fit(s3_input)


    # Deploy the model to hosting service





if __name__=='__main__':
    train_deploy_personalized_recommendation()
    sys.exit(0)
