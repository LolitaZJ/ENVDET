#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 21:51:36 2022

ENVDET is a pipline to achieve earthquake detection.

ENVDET consists of ENVnet and DETnet.

Train and Test ENVnet

Fast run!!!
#==========================================#

python ENVDET_ENV_MAIN.py --mode=train --epochs=3 --patience=1

python ENVDET_ENV_MAIN.py --mode=test

#==========================================# 
 

@author: zhangj2
"""
# In[]
#==========================================#
# Import libs
#==========================================# 
# common
import os
import pandas as pd
import numpy as np
import datetime
import argparse
# neural
import tensorflow as tf
import keras 
from keras import backend as K
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.models import Model,load_model
# custom
from model_master import ENV_model
from utils_master import plot_loss,DataGenerator_ENV,plot_env_res,env_dataset,save_tr_info,env_qc_dataset
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
                        help="model name")  
        
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
                        default="val_loss",
                        help="monitor the val_loss/loss/acc/val_acc")  
    
    parser.add_argument("--monitor_mode",
                        default="min",
                        help="min/max/auto") 
    
    parser.add_argument("--loss",
                        default='mse',
                        help="loss fucntion")  
    
    parser.add_argument("--use_multiprocessing",
                        default=False,
                        help="False,True")     
    
    parser.add_argument("--workers",
                        default=32,
                        type=int,
                        help="workers")     

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
    
    parser.add_argument("--rm_no",
                        default=True,
                        help="rm noise detected by STALTA")     
    
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

#==========================================#
# Save Configures
#==========================================# 
def set_configure(args):
    model_name=args.model_name
    wave_input=args.wave_input
    epochs=args.epochs
    patience=args.patience
    monitor=args.monitor
    mode=args.monitor_mode
    batch_size=args.batch_size
    use_multiprocessing=args.use_multiprocessing
    workers=args.workers
    loss=args.loss
    
    if not os.path.exists(args.conf_dir):
        os.mkdir(args.conf_dir)  
    # save configure
    f1 = open(args.conf_dir+'Conf_%s.txt'%args.model_name,'w')
    f1.write('Model: %s'%model_name+'\n')
    f1.write('epochs: %d'%epochs+'\n')
    f1.write('batch_size: %d'%batch_size+'\n')
    f1.write('monitor: %s'%monitor+'\n')
    f1.write('mode: %s'%mode+'\n')
    f1.write('patience: %d'%patience+'\n')
    f1.write('wave_input: %s'%str(wave_input)+'\n')
    f1.write('loss: %s'%loss+'\n')
    f1.write('workers: %d'%workers+'\n')
    f1.write('use_multiprocessing: %s'%use_multiprocessing+'\n')
    f1.close()
#==========================================#
# Main function
#==========================================#     
def main(args):
    if args.mode=='train':
        print('#==========================================#')
        print(args.mode)
        # Get data list
        ev_list_train,ev_list_validation,ev_list_test = env_dataset(args.csv_dir,args.no_dir,args.rm_no,args.weight)
        
        # Mk generator
        train_generator = DataGenerator_ENV(ev_list_train,args.data_dir, batch_size=args.batch_size)
        validation_generator = DataGenerator_ENV(ev_list_validation,args.data_dir, batch_size=args.batch_size)
        test_generator = DataGenerator_ENV(ev_list_test,args.data_dir, batch_size=args.batch_size)
        
        # Steps/Epoches
        steps_per_epoch=int(len(ev_list_train)//args.batch_size)
        validation_steps=int(len(ev_list_validation)//args.batch_size)
        
        # Build model
        model1=ENV_model(args.wave_input) 
        try:
            model1.compile(loss=args.loss,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        except:
            model1.compile(loss=args.loss,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        
        saveBestModel = ModelCheckpoint('./model/%s.h5'%args.model_name,monitor=args.monitor, verbose=1, 
                                        save_best_only=True, mode=args.monitor_mode)
        estop = EarlyStopping(monitor=args.monitor, patience=args.patience, verbose=0, mode=args.monitor_mode)
        callbacks_list = [saveBestModel,estop]
        
        # Fit 
        print('#==========================================#')
        print('Training~')        
        begin = datetime.datetime.now() 
        history_callback=model1.fit_generator(
                                              generator=train_generator, 
                                              steps_per_epoch= steps_per_epoch,                      
                                              epochs=args.epochs, 
                                              verbose=1,
                                              callbacks=callbacks_list,
                                              use_multiprocessing=args.use_multiprocessing,
                                              workers=args.workers,
        #                                     validation_split=0.1)
                                     validation_data=validation_generator,
                                     validation_steps=validation_steps)
                                    
        end = datetime.datetime.now()
        print(end-begin)
    
        # Plot and save acc & loss
        if not os.path.exists(args.acc_loss_fig):
            os.mkdir(args.acc_loss_fig)      
        plot_loss(history_callback,model=args.model_name,save_path=args.acc_loss_fig,c='k')
        # Save training info 
        file_path=args.conf_dir+'Conf_%s.txt'%args.model_name
        save_tr_info(history_callback,file_path)
        
        # Evaluate and predict
        env_model=load_model('./model/%s.h5'%args.model_name,custom_objects={'tf':tf})
        scores = env_model.evaluate_generator(generator=test_generator,
                                              workers=args.workers,
                                              use_multiprocessing=args.use_multiprocessing,
                                              verbose=1)   
        
        print('Testing accuracy: %.4f and loss: %.4f ' %(scores[1],scores[0]))
        f1 = open(file_path,'a+')
        f1.write('============evaluate==============='+'\n')
        f1.write('Testing accuracy: %.4f and loss: %.4f'%(scores[1],scores[0])+'\n')
        f1.close()
        # Save model weight
        env_model.save_weights('./model/%s_wt.h5'%args.model_name)

        
    if args.mode=='test':
        print('#==========================================#')
        print(args.mode)        
        env_model=load_model('./model/%s.h5'%args.model_name,custom_objects={'tf':tf})
        no1_list,no2_list,ev_10,ev_30,ev_60,ev_90,ev_120,ev_150=env_qc_dataset(args.csv_dir,args.no_dir,args.weight)

        save_path='./Res/%s/'%args.model_name
        if not os.path.exists(save_path):
            os.mkdir(save_path)         
        for ev_name in ['No_1','No_2','Ev_30','Ev_60','Ev_90','Ev_120','Ev_150']: 
            
            if ev_name=='No_1':
                p_list=[1,21,61,81,111,141,142,162]
                test_generator = DataGenerator_ENV(no1_list,args.data_dir, batch_size=256)
            if ev_name=='No_2':
                p_list=[2,22,101,151,182,201,251]
                test_generator = DataGenerator_ENV(no2_list,args.data_dir, batch_size=256)
                
            if ev_name=='Ev_30':
                p_list=[81,101,181,221]      
                test_generator = DataGenerator_ENV(ev_30,args.data_dir, batch_size=256)
            if ev_name=='Ev_60':
                p_list=[2,101,102,121,141,161,242]
                test_generator = DataGenerator_ENV(ev_60,args.data_dir, batch_size=256)
            if ev_name=='Ev_90':
                p_list=[31,102,121,142,202,211,242]  
                test_generator = DataGenerator_ENV(ev_90,args.data_dir, batch_size=256)
            if ev_name=='Ev_120':
                p_list=[63,81,201,241]  
                test_generator = DataGenerator_ENV(ev_120,args.data_dir, batch_size=256)
            if ev_name=='Ev_150':
                p_list=[124,184,201,221,243]  
                test_generator = DataGenerator_ENV(ev_150,args.data_dir, batch_size=256)
        
            tmp=iter(test_generator)
            tmp1=next(tmp)
            test_data=tmp1[0]['wave_input']
            test_label=tmp1[1]['env_output']
             
            pred=env_model.predict(test_data)         
                    
            for k in range(3):
                for i in p_list:  
                    n=256*k+i
                    plot_env_res(test_data,test_label,pred,n=n,save_path=save_path,k=k,name=ev_name)
                    print(i)    
    
# In[] main
if __name__ == '__main__':
    args = read_args()
    start_gpu(args)
    set_configure(args)
    main(args)
    
# In[]
