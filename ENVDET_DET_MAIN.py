#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 16:07:02 2022

ENVDET is a pipline to achieve earthquake detection.

ENVDET consists of ENVnet and DETnet.

Train and Test DETnet

Fast run!!!
#==========================================#
python ENVDET_DET_MAIN.py --mode=train --epochs=3 --patience=1

python ENVDET_DET_MAIN.py --mode=test
#==========================================#

@author: zhangj2
"""

# In[]
import os
import argparse
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import backend as K
import datetime
import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model,load_model
from sklearn.metrics import confusion_matrix

# function
from model_master import ENV_model,DET_model,bmodel,w_categorical_crossentropy,wrapped_partial
from utils_master import plot_confusion_matrix,plot_loss,DataGenerator_det,plot_env_res,det_dataset,save_tr_info

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
                        default="ENVDET_v1",
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
                        default="val_accuracy",
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

#==========================================#
# Save Configures
#==========================================# 
def set_configure(args):
    cl=args.cl
    model_name=args.model_name
    wave_input=args.wave_input
    epochs=args.epochs
    patience=args.patience
    monitor=args.monitor
    mode=args.monitor_mode
    batch_size=args.batch_size
    use_multiprocessing=args.use_multiprocessing
    workers=args.workers
    env_model_name='./model/%s_wt.h5'%args.env_name
    
    if not os.path.exists(args.conf_dir):
        os.mkdir(args.conf_dir)  
    # save configure
    f1 = open(args.conf_dir+'Conf_%s.txt'%args.model_name,'w')
    f1.write('Classes: %d'%cl+'\n')
    f1.write('Model: %s'%model_name+'\n')
    f1.write('epochs: %d'%epochs+'\n')
    f1.write('batch_size: %d'%batch_size+'\n')
    f1.write('monitor: %s'%monitor+'\n')
    f1.write('mode: %s'%mode+'\n')
    f1.write('patience: %d'%patience+'\n')
    f1.write('wave_input: %s'%str(wave_input)+'\n')
    f1.write('train: %.2f,validation: %.2f,test:%.2f '%(args.weight[0],args.weight[1],args.weight[2])+'\n')
    f1.write('workers: %d'%workers+'\n')
    f1.write('use_multiprocessing: %s'%use_multiprocessing+'\n')
    f1.write('env_model_name: %s'%env_model_name+'\n')
    f1.close()

#==========================================#
# Main function
#==========================================#     
def main(args):
    if args.mode=='train':
        print('#==========================================#')
        print(args.mode)
        # Get data list
        ev_list_train,ev_list_validation,ev_list_test,dis_cate = det_dataset(args.csv_dir,args.no_dir,args.rm_no,args.weight)

        # Mk generator
        train_generator = DataGenerator_det(ev_list_train,args.data_dir, batch_size=args.batch_size,nos=args.nos)
        validation_generator = DataGenerator_det(ev_list_validation,args.data_dir, batch_size=args.batch_size,nos=args.nos)
        test_generator = DataGenerator_det(ev_list_test,args.data_dir, batch_size=args.batch_size,nos=args.nos)
        
        steps_per_epoch=int(len(ev_list_train)//args.batch_size)
        validation_steps=int(len(ev_list_validation)//args.batch_size)

        # get the weights
        wts=np.ones((args.cl,args.cl))
        for i in range(args.cl):
            wts[i,i]=wts[i,i]/dis_cate[i]  
            
        # Build model
        env_model_name='./model/%s_wt.h5'%args.env_name
        model1=bmodel(args.wave_input,env_model_name,args.cl,wts)
        
        saveBestModel = ModelCheckpoint('./model/%s.h5'%args.model_name, monitor=args.monitor, verbose=1, 
                                        save_best_only=True,mode=args.monitor_mode)
        estop = EarlyStopping(monitor=args.monitor, patience=args.patience, verbose=0, mode=args.monitor_mode)
        callbacks_list = [saveBestModel,estop]
        
        # Fit
        begin = datetime.datetime.now() 
        history_callback=model1.fit_generator(
                                              generator=train_generator, 
                                              steps_per_epoch= steps_per_epoch,                      
                                              epochs=args.epochs, 
                                              verbose=1,
                                              callbacks=callbacks_list,
                                              # class_weight=weights,
                                              use_multiprocessing=args.use_multiprocessing,
                                              workers=args.workers,
        #                                     validation_split=0.1)
                                     validation_data=validation_generator,
                                     validation_steps=validation_steps)
                                    
        end = datetime.datetime.now()
        print(end-begin)

        # Plot and save loss accuracy
        if not os.path.exists(args.acc_loss_fig):
            os.mkdir(args.acc_loss_fig)      
        plot_loss(history_callback,model=args.model_name,save_path=args.acc_loss_fig,c='k')
        
        # Save training info 
        file_path=args.conf_dir+'Conf_%s.txt'%args.model_name
        save_tr_info(history_callback,file_path)   
        
    # if args.mode=='eva':         
    #     # Evaluate and predict
    #     ev_list_train,ev_list_validation,ev_list_test,dis_cate = det_dataset(args.csv_dir,args.no_dir,args.rm_no,args.weight)
    #     print(len(ev_list_test[:256*3]))
    #     # get the weights
    #     wts=np.ones((args.cl,args.cl))
    #     for i in range(args.cl):
    #         wts[i,i]=wts[i,i]/dis_cate[i]          
        
        # test_generator = DataGenerator_det(ev_list_test[:256*3],args.data_dir, batch_size=args.batch_size,nos=args.nos)
        model1=load_model('./model/%s.h5'%args.model_name,custom_objects={'w_categorical_crossentropy':w_categorical_crossentropy})
        ncce = wrapped_partial(w_categorical_crossentropy, weights=wts)
        model1.compile(loss=ncce,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

        scores = model1.evaluate_generator(generator=test_generator,
                                              workers=args.workers,
                                              use_multiprocessing=args.use_multiprocessing,
                                              verbose=1)   
        
        print('Testing accuracy: %.4f and loss: %.4f ' %(scores[1],scores[0]))
        f1 = open(file_path,'a+')
        f1.write('============evaluate==============='+'\n')
        f1.write('Testing accuracy: %.4f and loss: %.4f'%(scores[1],scores[0])+'\n')
        f1.close()
        
        # Save ENVDET model weight
        model1.save_weights('./model/%s_wt.h5'%args.model_name)
        # Save DETnet model weight
        ENV_m=Model(inputs=model1.get_layer('DET').input, outputs=model1.get_layer('DET').output)
        ENV_m.save_weights('./model/%s_wt.h5'%args.model_name1)
        ENV_m.compile(loss=args.loss,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        # Save DETnet model
        ENV_m.save('./model/%s.h5'%args.model_name1)

    if args.mode=='test':
        print('#==========================================#')
        print(args.mode)

        # Get data list
        ev_list_train,ev_list_validation,ev_list_test,dis_cate = det_dataset(args.csv_dir,args.no_dir,args.rm_no,args.weight)
        np.random.seed(0)
        np.random.shuffle(ev_list_test)        
        test_generator = DataGenerator_det(ev_list_test,args.data_dir, batch_size=args.batch_size,nos=args.nos)

        # Load model
        model1=load_model('./model/%s.h5'%args.model_name,custom_objects={'w_categorical_crossentropy':w_categorical_crossentropy})

        #  Predict all test data
        pred_y=[]
        val_y=[]
        k=0
        test_n=len(test_generator)
        for tmp in iter(test_generator):
            try:
                pred_y.append( model1.predict(tmp[0]['wave_input']))
                val_y.append(tmp[1]['DET'])
                k+=1
            except StopIteration:
                break
            if k==test_n:
                break
        
        pred_y=np.array(pred_y)
        val_y=np.array(val_y) 
        
        pred_y=pred_y.reshape(-1,args.cl)
        val_y=val_y.reshape(-1,args.cl)
        
        # Plt CM
        val_y1=np.argmax(val_y,axis=1)
        pred_y1=np.argmax(pred_y,axis=1)   
        print(val_y1[:10]) 
        confusion_mat = confusion_matrix(val_y1, pred_y1)
        np.savez('./Res/STEAD_test_CM_%s'%args.model_name1,mat=confusion_mat)
        
        plot_confusion_matrix(confusion_mat, classes=['Noise','<30 km','<60 km','<90 km','<120 km','>120 km'],
                              save_path=args.acc_loss_fig,name_m='CM_%s'%args.model_name1)
        # 
        file_path=args.conf_dir+'Conf_%s.txt'%args.model_name
        
        f1 = open(file_path,'a+')
        f1.write('===========precision================'+'\n')
        print('Precision:')
        for i in range(args.cl):
            su=np.sum(confusion_mat,axis=0)[i]
            if su>0:
                per=confusion_mat[i,i]/su
            else:
                per=0 
            print(i,per)
            print('%d: %.4f' %(i,per)+'\n')
            
        f1.write('===========Recall==============='+'\n')    
        print('Recall:')
        for i in range(args.cl): 
            su=np.sum(confusion_mat,axis=1)[i]
            if su>0:
                per=confusion_mat[i,i]/su
            else:
                per=0
            print(i,per)
            print('%d: %.4f' %(i,per)+'\n')
        f1.close()
        
        # TP FP TN FN
        TP_noise_index=[]
        FP_noise_index=[]
        TN_noise_index=[]
        FN_noise_index=[]
        
        for i in range(len(pred_y1)):
            if pred_y1[i]==0 and val_y1[i]==0:
                TP_noise_index.append(i)   
            if pred_y1[i]==0 and val_y1[i]!=0:
                FP_noise_index.append(i)        
            if pred_y1[i]!=0 and val_y1[i]!=0:
                TN_noise_index.append(i)               
            if pred_y1[i]!=0 and val_y1[i]==0:
                FN_noise_index.append(i) 
                
        # Precision recall
        try:
            precesion_noise=len(TP_noise_index)/(len(TP_noise_index)+len(FP_noise_index))
        except:
            precesion_noise=0
        try:    
            recall_noise=len(TP_noise_index)/(len(TP_noise_index)+len(FN_noise_index))
        except:
            recall_noise=0
        try:
            precesion_event=len(TN_noise_index)/(len(TN_noise_index)+len(FN_noise_index))
        except:
            precesion_event=0
        try:
            recall_event=len(TN_noise_index)/(len(TN_noise_index)+len(FP_noise_index))
        except:
            recall_event=0
        
        Acc=(len(TP_noise_index)+len(TN_noise_index))/(len(TN_noise_index)+len(FP_noise_index)+len(TP_noise_index)+len(FN_noise_index))
        
        print('TP_noise: %d' %len(TP_noise_index))
        print('FP_noise: %d' %len(FP_noise_index))
        
        print('TN_noise: %d' %len(TN_noise_index))
        print('FN_noise: %d' %len(FN_noise_index))
        
        print('Precision_noise: %.4f' %precesion_noise)
        print('Recall_noise: %.4f' %recall_noise)
        
        print('Precision_event: %.4f' %precesion_event)
        print('Recall_event: %.4f' %recall_event)
        
        print('Accuracy: %.4f' %Acc)
        
        f1 = open(file_path,'a+')
        f1.write('============TPFPTNFN==============='+'\n')
        f1.write('TP_noise: %d' %len(TP_noise_index)+'\n')
        f1.write('FP_noise: %d' %len(FP_noise_index)+'\n')
        f1.write('TN_noise: %d' %len(TN_noise_index)+'\n')
        f1.write('FN_noise: %d' %len(FN_noise_index)+'\n')
        
        f1.write('Precision_noise: %.4f' %precesion_noise+'\n')
        f1.write('Recall_noise: %.4f' %recall_noise+'\n')
        f1.write('Precision_event: %.4f' %precesion_event+'\n')
        f1.write('Recall_event: %.4f' %recall_event+'\n')
        f1.write('Accuracy: %.4f' %Acc+'\n')
        
        f1.close()  
          
# In[] main
if __name__ == '__main__':
    args = read_args()
    start_gpu(args)
    set_configure(args)
    main(args)        
        
# In[]        
        


        
        
    

