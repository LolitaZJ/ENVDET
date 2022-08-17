#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:36:52 2022

ENVDET is a pipline to achieve earthquake detection.

Test and Predict ENVDET

Fast run!!!
#==========================================#

python ENVDET_MAIN.py --mode=test --save_result

python ENVDET_MAIN.py --mode=predict --save_result

#==========================================# 

@author: zhangj2
"""

# In[]
import os
import argparse
import datetime

import tensorflow as tf
from keras import backend as K
import keras
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.models import Model,load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# function
from utils_master import env_dataset
from utils_master import DataGenerator_ENV,plot_ENVDET,DataGenerator_ENVDET
import model_master

# plot
import matplotlib.pyplot as plt
import matplotlib  
matplotlib.use('Agg') 

# In[]
#==========================================#
# Set GPU
#==========================================# 
def start_gpu(args):
    cuda_kernel=args.GPU
    os.getcwd()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_kernel
    
    try:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print('Physical GPU：', len(gpus))
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print('Logical GPU：', len(logical_gpus))
    except:
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
        session = tf.Session(config=config)
        
#==========================================#
# Set Configures
#==========================================# 
def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU",
                        default="3",
                        help="set gpu ids") 
    
    parser.add_argument("--mode",
                        default="train",
                        help="/train/test/predict/")
    
    parser.add_argument("--model_name",
                        default="ENVnet_v1",
                        help="ENVDET pipline") 
    
    parser.add_argument("--model_name1",
                        default="DETnet_v1",
                        help="DETnet model name")     
    
    parser.add_argument("--env_name",
                        default="ENVnet_v1",
                        help="ENVnet model name")     
    

    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help="number of epochs (default: 100)")
    
    parser.add_argument("--batch_size",
                        default=256,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="learning rate")
    
    parser.add_argument("--patience",
                        default=10,
                        type=int,
                        help="early stopping")
    
    parser.add_argument("--clas",
                        default=3,
                        type=int,
                        help="number of class") 
    
    parser.add_argument("--monitor",
                        default="val_acc",
                        help="monitor the val_loss/loss/acc/val_acc")  
    
    parser.add_argument("--monitor_mode",
                        default="max",
                        help="min/max/auto") 
    
    parser.add_argument("--loss",
                        default='categorical_crossentropy',
                        help="loss fucntion")  
    
    parser.add_argument("--use_multiprocessing",
                        default=False,
                        help="False,True")     
    
    parser.add_argument("--workers",
                        default=32,
                        type=int,
                        help="workers")   
    
    parser.add_argument("--cl",
                        default=6,
                        type=int,
                        help="classes")      
    
    parser.add_argument("--model_dir",
                        default='./model/',
                        help="Checkpoint directory")

    parser.add_argument("--num_plots",
                        default=10,
                        type=int,
                        help="Plotting trainning results")
    
    parser.add_argument("--wave_input",
                        default=(6000,1,3),
                        type=int,
                        help="wave_input")
        
    parser.add_argument("--data_dir",
                        default="./data/merged.hdf5",
                        help="Input data file directory")
    
    parser.add_argument("--no_dir",
                        default="./data/stalta_pk_noise.npz",
                        help="Input data file directory")
    
    parser.add_argument("--csv_dir",
                        default="./data/merged.csv",
                        help="Input csv file directory")  
    
    parser.add_argument("--pred_dir",
                        default="./data/HINET_Polarity.hdf5",
                        help="Input data file directory")
    
    parser.add_argument("--pred_csv",
                        default="./data/HINET_Polarity.csv",
                        help="Input csv file directory")      
    
    parser.add_argument("--pred_name",
                        default="Hinet_data",
                        help="predict data name")     

    parser.add_argument("--rm_no",
                        default=True,
                        help="rm noise detected by STALTA") 
    
    parser.add_argument("--nos",
                        default=False,
                        help="use augmentation")    

    parser.add_argument("--weight",
                        default=[0.85,0.10,0.05],
                        type=list,
                        help="weights of train,test,and validation") 
    
    parser.add_argument("--output_dir",
                        default='./Res/',
                        help="Output directory")
    
    parser.add_argument("--conf_dir",
                        default='./model_configure/',
                        help="Configure directory")    
    
    parser.add_argument("--acc_loss_fig",
                        default='./acc_loss_fig/',
                        help="acc&loss directory")    

    parser.add_argument("--plot_figure",
                        action="store_true",
                        help="If plot figure for test")
    
    parser.add_argument("--save_result",
                        action="store_true",
                        help="If save result for test")
           
    args = parser.parse_args()
    return args


# In[] load model
def main(args):
    env_model=load_model('./model/%s.h5'%args.env_name,custom_objects={'tf':tf})
    det_model=load_model('./model/%s.h5'%args.model_name1,custom_objects={'tf':tf})
    envdet=load_model('./model/%s.h5'%args.model_name,custom_objects={'tf':tf})
    
    if args.mode=='test':
        ev_list_train,ev_list_validation,ev_list_test = env_dataset(args.csv_dir,args.no_dir,args.rm_no,args.weight)
        
        # shuffle
        np.random.seed(0)
        np.random.shuffle(ev_list_test)
        
        test_generator = DataGenerator_ENV(ev_list_test,args.data_dir,batch_size=args.batch_size,ss=1)
        
        tmp=iter(test_generator)
        tmp1=next(tmp)
        test_data=tmp1[0]['wave_input']
        test_label=tmp1[1]['env_output']
    
        # Predict STEAD test data 
        pred=env_model.predict(test_data)
        pred1=det_model.predict(pred)
        
        if args.save_result:
            save_path=args.output_dir+args.pred_name+'/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            for i in range(0,args.batch_size,100):
                for k in range(3):
                    plot_ENVDET(test_data,test_label,pred,pred1,c=i,save_path=save_path,k=k)
                    print(i)
                    
    if args.mode=='predict':  
        
        Hi_csv=pd.read_csv(args.pred_csv)
        list_hi = Hi_csv['trace_name'].tolist()

        pred_gen=DataGenerator_ENVDET(list_hi,args.pred_dir,batch_size=args.batch_size)
        # ==== #
        #envdata = env_model.predict_generator(generator=pred_gen)
        #detdata = det_model.predict(envdata)
        # ==== #
        det=envdet.predict_generator(generator=pred_gen)
        if args.save_result:
            np.savez(args.output_dir+args.pred_name,det=det)

 # In[] main
if __name__ == '__main__':
    args = read_args()
    start_gpu(args)
    main(args)                
            
                