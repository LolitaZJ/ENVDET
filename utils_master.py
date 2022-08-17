#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 11:39:01 2022

@author: zhangj2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:07:25 2021

@author: zhangj
"""

import numpy as np
import math
import h5py
import pandas as pd
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

from keras.models import Model,load_model
#envdet_env=load_model('/data/zhangj/SELSS/envlope3_4.fcn')
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy import signal
import os
from obspy.signal.trigger import trigger_onset
import random
import itertools
from sklearn.metrics import confusion_matrix
import keras 
import tensorflow as tf
from obspy.signal.trigger import recursive_sta_lta,classic_sta_lta,trigger_onset
import pandas as pd
import math
# In[]
try:
    from keras.utils import Sequence
except:
    from tensorflow.keras.utils import Sequence

    
# In[] load ENVnet dataset
def env_dataset(csv_file,no_file,rm_no,weight):

    # get STEAD data
    df = pd.read_csv(csv_file)
    # shuffle
    df = df.sample(frac=1,random_state=0)
    # separate
    df1 = df[(df.trace_category == 'noise')] 
    df2 = df[(df.trace_category == 'earthquake_local')] 
    # train 0.85 test 0.15 validation 0.05
    w1=weight[0]
    w2=weight[0]+weight[1]
    
    ev_list_train = df1.iloc[0:int(len(df1)*w1)]['trace_name'].tolist() + df2.iloc[0:int(len(df2)*w1)]['trace_name'].tolist()
    ev_list_validation = df1.iloc[int(len(df1)*w2):]['trace_name'].tolist()+df2.iloc[int(len(df2)*w2):]['trace_name'].tolist()
    ev_list_test = df1.iloc[int(len(df1)*w1):int(len(df1)*w2)]['trace_name'].tolist() +df2.iloc[int(len(df2)*w1):int(len(df2)*w2)]['trace_name'].tolist()
    
    # remove the detect noise by STALTA
    if rm_no:
        L=np.load(no_file) # get noise by STALTA
        pk_no=L['pk_no']   
        ev_list_train=list(set(ev_list_train).difference(set(pk_no)))
        ev_list_validation =list(set(ev_list_validation).difference(set(pk_no)))
        ev_list_test=list(set(ev_list_test).difference(set(pk_no)))
    
    return ev_list_train,ev_list_validation,ev_list_test

# In[] QC the testing dataset
def env_qc_dataset(csv_file,no_file,weight):

    df = pd.read_csv(csv_file)
    # shuffle
    df = df.sample(frac=1,random_state=0)
    # separate
    df1 = df[(df.trace_category == 'noise')] 
    df2 = df[(df.trace_category == 'earthquake_local')] 
    
    w1=weight[0]
    w2=weight[0]+weight[1]
    
    # detect noise
    L=np.load(no_file)
    # testing noise dataset 
    pk_no=L['pk_no']
    no_list= df1.iloc[int(len(df1)*w1):int(len(df1)*w2)]['trace_name'].tolist() 
    no1_list=list(set(no_list).intersection(set(pk_no))) #  include 
    no2_list=list(set(no_list).difference(set(pk_no))) # except detected noise
    # testing event dataset ordering by distance
    df3=df2.iloc[int(len(df2)*0.85):int(len(df2)*0.95)]
    ev_10=df3[df3.source_distance_km<10]['trace_name'].tolist()
    ev_30=df3[(df3.source_distance_km>10) & (df3.source_distance_km<30)]['trace_name'].tolist() 
    ev_60=df3[ (df3.source_distance_km>30) & (df3.source_distance_km<60) ]['trace_name'].tolist() 
    ev_90=df3[ (df3.source_distance_km>60) & (df3.source_distance_km<90) ]['trace_name'].tolist() 
    ev_120=df3[ (df3.source_distance_km>90) & (df3.source_distance_km<120) ]['trace_name'].tolist() 
    ev_150=df3[ (df3.source_distance_km>120)]['trace_name'].tolist() 
    
    return no1_list,no2_list,ev_10,ev_30,ev_60,ev_90,ev_120,ev_150



# In[] Save training info 
def save_tr_info(history_callback,file_path):
    history_dict=history_callback.history
    loss_value=history_dict['loss'] 
    val_loss_value=history_dict['val_loss']
    acc_value=history_dict['accuracy']
    val_acc_value=history_dict['val_accuracy']
    
    best_inx=np.argmin(val_loss_value)
    
    f1 = open(file_path,'a+')
    f1.write('============training==============='+'\n')
    f1.write('loss_value\n')
    f1.write('%s\n'%(str(loss_value)))
    f1.write('val_loss_value\n')
    f1.write('%s\n'%(str(val_loss_value)))
    f1.write('acc_value\n')
    f1.write('%s\n'%(str(acc_value)))
    f1.write('val_acc_value\n')
    f1.write('%s\n'%(str(val_acc_value)))
    f1.write('Training accuracy: %.4f and loss: %.4f'%(acc_value[best_inx],loss_value[best_inx])+'\n')
    f1.write('Validation accuracy: %.4f and loss: %.4f'%(val_acc_value[best_inx],val_loss_value[best_inx])+'\n')
    f1.close()


# In[] plot testing result
def plot_env_res(st_data,env_data,pred,n=0,save_path=None,k=0,name='No'):
    font1 = {'family' : 'Times New Roman','weight' : 'bold','size':18}
    font2 = {'family' : 'Times New Roman','weight' : 'bold','size': 20}
    
    figure, ax = plt.subplots(3,1,figsize=(9,9))
    ax[0].plot(st_data[n,:,0,0],'k',label='Waveform')
    ax[0].plot(env_data[n,:,0,0],'b',label='True')
    ax[0].plot(pred[n,:,0,0],'r',label='Predicted')
    ax[0].set_ylim([-1,1])
    ax[0].set_xlim([6,6000])
    ax[0].tick_params(labelsize=22)
    labels = ax[0].get_xticklabels() + ax[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax[0].legend(prop=font1,bbox_to_anchor=(1.0, 1.3),frameon=False,ncol=3)
    ax[0].text(6100, 0.6, 'E',family='Times New Roman',fontsize=15)
    
    ax[1].plot(st_data[n,:,0,1],'k')
    ax[1].plot(env_data[n,:,0,1],'b')
    ax[1].plot(pred[n,:,0,1],'r')
    ax[1].set_ylabel('Amplitude',font2)
    ax[1].set_ylim([-1,1])
    ax[1].set_xlim([6,6000])
    ax[1].tick_params(labelsize=22)
    labels = ax[1].get_xticklabels() + ax[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax[1].text(6100, 0.6, 'N',family='Times New Roman',fontsize=15)
    
    ax[2].plot(st_data[n,:,0,2],'k')
    ax[2].plot(env_data[n,:,0,2],'b')
    ax[2].plot(pred[n,:,0,2],'r')
    ax[2].set_xlabel('Samples',font2)
    ax[2].set_ylim([-1,1])
    ax[2].set_xlim([6,6000])
    ax[2].tick_params(labelsize=22)
    labels = ax[2].get_xticklabels() + ax[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax[2].text(6100, 0.6, 'Z',family='Times New Roman',fontsize=15)
    if not save_path is None:
        plt.savefig(save_path+'ENV_%s_%d_%d'%(name,k,n),dpi=600)
        np.savez(save_path+'ENV_%s_%d_%d'%(name,k,n), data=st_data[n,:,0,:],label=env_data[n,:,0,:],pred=pred[n,:,0,:])
    plt.show()    
    
    # In[]
def plot_loss(history_callback,model='model',save_path=None,c='c'):
    font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

    history_dict=history_callback.history
    
    loss_value=history_dict['loss'] 
    val_loss_value=history_dict['val_loss']
    acc_value=history_dict['accuracy']
    val_acc_value=history_dict['val_accuracy']
    epochs=range(1,len(loss_value)+1)
    if not save_path is None:
     np.savez(save_path+'acc_loss_%s'%model,
              loss=loss_value,val_loss=val_loss_value,
              acc=acc_value,val_acc=val_acc_value)

    figure, ax = plt.subplots(figsize=(8,6))
    if c=='k':
        tmp=[]
        tmp=acc_value.copy()
        print(tmp)
        tmp.insert(0,0)
        print(tmp)
        print(len(tmp))
        plt.plot(np.arange(len(tmp)),tmp[:],'k',label='Training acc')
        tmp1=[]
        tmp1=val_acc_value.copy()
        tmp1.insert(0,0)
        plt.plot(np.arange(len(tmp1)),tmp1[:],'k-.',label='Validation acc')  
        
    else:
        plt.plot(epochs,acc_value,'r',label='Training acc')
        plt.plot(epochs,val_acc_value,'b-.',label='Validation acc')

    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # plt.ylim([0.75,0.95])
    plt.xlabel('Epochs',font2)
    plt.ylabel('Accuracy',font2)
    plt.legend(prop=font2,loc='lower right')    
    if not save_path is None:
        if c=='k':
            plt.savefig(save_path+'Acc_%s_%s.png'%(model,c),dpi=600)    
        else:        
            plt.savefig(save_path+'Acc_%s.png'%model,dpi=600)  
    plt.show()
    
    figure, ax = plt.subplots(figsize=(8,6))
    if c=='k':
        plt.plot(epochs,loss_value,'k',label='Training loss')
        plt.plot(epochs,val_loss_value,'k-.',label='Validation loss')        
    else: 
        plt.plot(epochs,loss_value,'r',label='Training loss')
        plt.plot(epochs,val_loss_value,'b-.',label='Validation loss')
    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]    
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)  
    plt.legend(prop=font2)
    if not save_path is None:
        if c=='k':
            plt.savefig(save_path+'Loss_%s_%s.png'%(model,c),dpi=600)    
        else:
            plt.savefig(save_path+'Loss_%s.png'%model,dpi=600)      
    plt.show()   
    
    
    
  
# In[] load DETnet dataset
def det_dataset(csv_file,no_file,rm_no,weight):    

    df = pd.read_csv(csv_file)
    # shuffle
    df = df.sample(frac=1,random_state=0)
    # separate
    df1 = df[(df.trace_category == 'noise')] 
    df2 = df[(df.trace_category == 'earthquake_local')] 
    # detail
    df_30=df2[(df2.source_distance_km<30)]
    df_60=df2[(df2.source_distance_km>=30) & (df2.source_distance_km<60)]
    df_90=df2[(df2.source_distance_km>=60) & (df2.source_distance_km<90)]
    df_120=df2[(df2.source_distance_km>=90) & (df2.source_distance_km<120)]
    df_150=df2[(df2.source_distance_km>=120)]
    dis_cate=[len(df1),len(df_30),len(df_60),len(df_90),len(df_120),len(df_150) ]
    # weight
    w1=weight[0]
    w2=weight[0]+weight[1]
    # train 0.85 test 0.15 validation 0.05
    ev_list_train = df1.iloc[0:int(len(df1)*w1)]['trace_name'].tolist() + df2.iloc[0:int(len(df2)*w1)]['trace_name'].tolist()
    ev_list_validation = df1.iloc[int(len(df1)*w2):]['trace_name'].tolist()+df2.iloc[int(len(df2)*w2):]['trace_name'].tolist()
    ev_list_test = df1.iloc[int(len(df1)*w1):int(len(df1)*w2)]['trace_name'].tolist() +df2.iloc[int(len(df2)*w1):int(len(df2)*w2)]['trace_name'].tolist()
    
    # rm detected noise
    if rm_no:
        L=np.load(no_file) #detected noise
        pk_no=L['pk_no']
        dis_cate=[len(df1)-len(pk_no),len(df_30),len(df_60),len(df_90),len(df_120),len(df_150) ]
        ev_list_train=list(set(ev_list_train).difference(set(pk_no)))
        ev_list_validation =list(set(ev_list_validation).difference(set(pk_no)))
        ev_list_test=list(set(ev_list_test).difference(set(pk_no)))
        
    dis_cate=dis_cate/np.min(dis_cate)
    
    return ev_list_train,ev_list_validation,ev_list_test,dis_cate


# In[]
#  20220723
class DataGenerator_ENV(Sequence):

    def __init__(self, list_IDs,file_name,batch_size=256,db_no=[5,20],shuffle=True,ss=0):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.file_name = file_name
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.db_no=db_no
        self.ss=ss

    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
            
        # get batch data index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]        
        
        # read batch data
        X, Y= self._read_data(list_IDs_temp,indexes[0])

        return ({'wave_input': X}, {'env_output':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise

    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2    

    def _gen_label(self,itp,its):
        target = np.zeros((6000,1,2))
        if itp>target.shape[0] or itp<0:
            pass  
        target[int(itp):int(its), 0, 0]=1
    
        ite=int(itp+3*(its-itp))
        if ite>6000:
            ite=6000  
        target[int(its):int(ite), 0, 1]=1
        return target 
    
    def _add_spike(self,data1,y1,k):
        np.random.seed(k)
        n1=np.random.randint(0,6000)
        n2=np.random.randint(0,6000)
        f1=np.min([n1,n2])
        f2=np.max([n1,n2])
        n3=np.random.randint(1,20) #13
        if n3<10:
            if f2!=f1:
                no1=np.load('/data2/zhangj2/ENVDET_MASTER_38/data/special_noise1.npz')['data']
                if np.max(abs(no1[f1:f2,2]))>0.5: 
                    n4=np.random.randint(0,6000-f2+f1)
                    nosie=np.zeros(len(data1),)
                    nosie[n4:n4+f2-f1]=no1[f1:f2,2]
                    data1=data1*n3/10+nosie
                    y1=y1*n3/10
        return data1,y1
        
    def _add_zeros(self,data1,y1,k):
        np.random.seed(k) 
        n3=np.random.randint(0,4)
        if n3<1:
                n1=np.random.randint(500,2000)
                n4=np.random.randint(100,6000-n1-100)
                data1[n4:n4+n1]=0.0
                y1[n4:n4+n1]=(np.mean(y1[:n4])+np.mean(y1[n4+n1:]))/2

        return data1,y1 
      
    def _smooth(self,y, box_pts=1000):
        box = np.ones(box_pts) / box_pts
        y_smooth1 = np.convolve(y, box, mode='valid')
        y_smooth2 = np.convolve(y[::-1], box, mode='valid')
        y_smooth=np.hstack((y_smooth1,y_smooth2[:box_pts-1]))
        return y_smooth    

    def _inte_env(self,data1,d,tp1,fl=0,index=0,k=0,ss=0):
        
        font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
        
        if tp1>0:
            ln=len(data1)
            data2=data1.reshape(int(ln/d),d)
            data2=np.max(data2,axis=1)
        
            x=np.linspace(0,ln,int(ln/d))
            y=data2
            f=interpolate.interp1d(x,abs(y),kind='cubic')
            x1=np.linspace(0,ln,ln)
            y1=f(x1)
            y1[y1<0.01]=0.01

            wed=1500
            y1[0:int(tp1)]=0.01
            if index==0:
                index=int(np.argmax(y1))
            else:
                index=int(index)
                wed=int(index-tp1)*3
                
            ed_index=index+wed
                
            if ed_index>ln:
                ed_index=6000
                
            weg=np.ones((ln,))
            weg[index:]=0
            weg[index:ed_index]=-1/wed*(np.arange(index,ed_index)-index)+1
            y1=y1*weg
        else:
            data2=np.zeros((6000,))
            data2=data1.copy()
            
            for i in range(2):
                std_data = np.std(data2)
                data2[abs(data2)>=std_data*3]=0

            ln=len(data2)
            data3=data2.reshape(int(ln/d),d)
            data3=np.max(data3,axis=1)            
            x=np.linspace(0,ln,int(ln/d))
            y=data3
            f=interpolate.interp1d(x,abs(y),kind='cubic')
            x1=np.linspace(0,ln,ln)
            y1=f(x1)
            
            if np.mean(abs(y1))<0.05:
                y1[:]=np.mean(abs(y1))

            y1[y1<0]=0.01           

            if ss==1:  
                data1,y1=self._add_spike(data1,y1,k) 
                data1,y1=self._add_zeros(data1,y1,k)
                

        if fl==1:
            figure, ax = plt.subplots(figsize=(6,4) )  
            t=np.arange(6000)*0.01
            plt.plot(t,data1)
            plt.plot(x*0.01,y)
            plt.plot(t,y1)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            plt.xlabel('Time (s)',font2) 
            plt.ylabel('Amplitude',font2)
            plt.show()
       
        return y1,data1
        
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1  

    def _stalta(self,st_data,ln=6,sn=2,re=False):
        df=100
        det=[]
        cft1=[]
        # on_det=[]
        for j in range(3):
            tmp=st_data[:,j]
            tmp=bp_filter(tmp,2,1,45,dt=0.01)
            cft = recursive_sta_lta(tmp, int(2 * df), int(6 * df))
            cft1.append(cft)
            det.append(np.max(cft))

        onset_index=[i for i in det if i>2.1]
        is_event = len(onset_index)>0

        if re:
            return is_event,np.array(cft1).transpose(1,0)
        else:
            return is_event

    def _read_data(self, list_IDs_temp,c1):

        batch_size=self.batch_size
        train_x=np.zeros((batch_size,6000,3))
        train_x2=np.zeros((batch_size,6000,3))
        train_x3=np.zeros((batch_size,6000,3))
        
        pt=np.zeros((batch_size,))  
        st=np.zeros((batch_size,))
        train_y=np.zeros((batch_size,6000,1,3))
        train_y2=np.zeros((batch_size,6000,1,3))
        train_y3=np.zeros((batch_size,6000,1,3))
        
        fl = h5py.File(self.file_name, 'r')
        #------------------------#    
        for c, evi in enumerate(list_IDs_temp):
            dataset = fl.get('data/'+str(evi))
            if evi.split('_')[-1] == 'NO':
                pt[c] = -1
                st[c] = -2
            else:
                try:
                    pt[c] = dataset.attrs['p_arrival_sample']
                    st[c] = dataset.attrs['s_arrival_sample']  
                except:
                    pt[c] = -1
                    st[c] = -2
                    print(evi)

            if pt[c] <=0:
                try:
                    tmp = np.array(dataset)
                except:
                    tmp = np.zeros((6000,3))
                    print(evi)
                train_x[c,:,:3]=tmp             
            else:
                train_x[c,:,:3] = np.array(dataset) 
                
            # bandpass (1-5) (20-45)
            hp=random.randint(20,45) 
            hp1=random.randint(1,5) 
            sig_data1=np.zeros((3,6000)) # open new address
            sig_data1=train_x[c,:,:3].transpose(1,0)
            sig_data1 = self._bp_filter(sig_data1,2,hp1,hp,0.01)
            sig_data1 = self._normal3(sig_data1.transpose(1,0).reshape(1,6000,3))
            train_x2[c,:,:3]=sig_data1[0,:,:]
            
            # bandpass 1-45
            temp=(train_x[c,:,:3]).transpose(1,0)
            temp=self._bp_filter(temp,2,1,45,0.01)
            temp=self._normal3(temp.transpose(1,0).reshape(1,6000,3))
            
            for ii in range(3):
                train_y[c,:,0,ii],train_x[c,:,ii]=self._inte_env(temp[0,:,ii],100,pt[c],index=st[c],k=c+c1,ss=self.ss)
                    
            for ii in range(3):
                train_y2[c,:,0,ii],train_x2[c,:,ii]=self._inte_env(train_x2[c,:,ii],100,pt[c],index=st[c],k=c+c1+1,ss=self.ss)

            train_x3[c,:,:3]=train_x2[c,:,:3].copy()
            train_y3[c,:,0,:3]=train_y2[c,:,0,:3].copy()
            
            maxn=np.argmax(abs(train_x2[c,:,2]))
            cc=np.random.randint(0,3)
            
            cf=np.random.randint(1,5)
            cf1=np.random.randint(1,5)
            ed=int(maxn+cf*100)
            if ed>5800:
                ed=5800
            star=int(maxn-cf1*100)
            if star<200:
                star=200
            if star<ed:
                try:    
                    if cc==0:
                        train_x3[c,:ed-star,:]=train_x3[c,star:ed,:3]
                        train_y3[c,:ed-star,0,:]=train_y2[c,star:ed,0,:3]
                    if cc==1:
                        train_x3[c,star-ed:,:]=train_x3[c,star:ed,:3]
                        train_y3[c,star-ed:,0,:]=train_y2[c,star:ed,0,:3]
                    if cc==2:
                        train_x3[c,star-ed:,:]=train_x3[c,star:ed,:3]
                        train_y3[c,star-ed:,0,:]=train_y2[c,star:ed,0,:3]
                        ed=int(maxn+cf1*100)
                        if ed>5800:
                            ed=5800
                        star=int(maxn-cf*100)
                        if star<200:
                            star=200
                        if star<ed:
                            train_x3[c,:ed-star,:]=train_x3[c,star:ed,:3]
                            train_y3[c,:ed-star,0,:]=train_y2[c,star:ed,0,:3]
                except:
                    print(cc)
                    print(ed-star,star,ed)                    
                    
        fl.close()
        train_x=self._normal3(train_x)   
        train_x=train_x.reshape(batch_size,6000,1,3)
        train_x2=self._normal3(train_x2)
        train_x2=train_x2.reshape(batch_size,6000,1,3)
        train_x3=self._normal3(train_x3)
        train_x3=train_x3.reshape(batch_size,6000,1,3)
        
        return np.vstack([train_x,train_x2,train_x3]), np.vstack([train_y,train_y2,train_y3])        

# In[]
class DataGenerator_ENV4(Sequence):

    def __init__(self, list_IDs,file_name,batch_size=256,db_no=[5,20],shuffle=True):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.file_name = file_name
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.db_no=db_no

    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
            
        # get batch data index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]        
        
        # read batch data
        X, Y= self._read_data(list_IDs_temp,indexes[0])

        return ({'wave_input': X}, {'env_output':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            x_max=np.max(abs(data1))
            if x_max!=0.0:
                data2[i,:,:]=data1/x_max 
        return data2
        
    def _gen_label(self,itp,its):
        target = np.zeros((6000,1,2))
        if itp>target.shape[0] or itp<0:
            pass  
        target[int(itp):int(its), 0, 0]=1
    
        ite=int(itp+3*(its-itp))
        if ite>6000:
            ite=6000  
        target[int(its):int(ite), 0, 1]=1
        return target    
    
    def _add_spike(self,data1,y1,k):
        np.random.seed(k)
        n1=np.random.randint(0,6000)
        n2=np.random.randint(0,6000)
        f1=np.min([n1,n2])
        f2=np.max([n1,n2])
        n3=np.random.randint(1,20) #13
        if n3<10:
            if f2!=f1:
                no1=np.load('/data2/zhangj2/ENVDET_MASTER_38/data/special_noise1.npz')['data']
                if np.max(abs(no1[f1:f2,2]))>0.5: 
                    n4=np.random.randint(0,6000-f2+f1)
                    nosie=np.zeros(len(data1),)
                    nosie[n4:n4+f2-f1]=no1[f1:f2,2]
                    data1=data1*n3/10+nosie
                    y1=y1*n3/10
        return data1,y1
    
    
    def _add_zeros(self,data1,y1,k):
        np.random.seed(k) 
        n3=np.random.randint(0,4)
        if n3<1:
                n1=np.random.randint(500,2000)
                n4=np.random.randint(100,6000-n1-100)
                data1[n4:n4+n1]=0.0
                y1[n4:n4+n1]=(np.mean(y1[:n4])+np.mean(y1[n4+n1:]))/2

        return data1,y1 
      
    def _smooth(self,y, box_pts=1000):
        box = np.ones(box_pts) / box_pts
        y_smooth1 = np.convolve(y, box, mode='valid')
        y_smooth2 = np.convolve(y[::-1], box, mode='valid')
        y_smooth=np.hstack((y_smooth1,y_smooth2[:box_pts-1]))
        return y_smooth    

    def _inte_env(self,data1,d,tp1,fl=0,index=0,k=0,ss=1):
        
        font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
        
        ln=len(data1)
        data2=data1.reshape(int(ln/d),d)
        data2=np.max(data2,axis=1)
    
        x=np.linspace(0,ln,int(ln/d))
        y=data2
        f=interpolate.interp1d(x,abs(y),kind='cubic')
        x1=np.linspace(0,ln,ln)
        y1=f(x1)
        y1[y1<0]=0
        if tp1>0:
            # y1=self._smooth(y1, box_pts=200)
            wed=1500
            y1[0:int(tp1)]=0
            if index==0:
                index=int(np.argmax(y1))
            else:
                index=int(index)
                wed=int(index-tp1)*2
                
            ed_index=index+wed
                
            if ed_index>ln:
                ed_index=6000
                
            weg=np.ones((ln,))
            weg[index:]=0
            weg[index:ed_index]=-1/wed*(np.arange(index,ed_index)-index)+1
            y1=y1*weg
        else:
            if ss==1:
                # y1=self._smooth(y1, box_pts=200)
                data1,y1=self._add_spike(data1,y1,k)    
                data1,y1=self._add_zeros(data1,y1,k)
            # if ss==-1:
            #     y1=self._smooth(y1, box_pts=200)
        if fl==1:
            figure, ax = plt.subplots(figsize=(6,4) )  
            t=np.arange(6000)*0.01
            plt.plot(t,data1)
            plt.plot(x*0.01,y)
            plt.plot(t,y1)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            plt.xlabel('Time (s)',font2) 
            plt.ylabel('Amplitude',font2)
            plt.show()
       
        return y1,data1
        
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1      
    
    def _read_data(self, list_IDs_temp,c1):
        """Read a batch data.
        ---
        # Arguments
            batch_files: the file of batch data.

        # Returns
            images: (batch_size, (image_shape)).
            labels: (batch_size, (label_shape)). """
        batch_size=self.batch_size
        train_x=np.zeros((batch_size,6000,3))
        train_x2=np.zeros((batch_size,6000,3))
        train_x3=np.zeros((batch_size,6000,3))
        
        pt=np.zeros((batch_size,))  
        st=np.zeros((batch_size,))
        train_y=np.zeros((batch_size,6000,1,3))
        train_y2=np.zeros((batch_size,6000,1,3))
        train_y3=np.zeros((batch_size,6000,1,3))
        
        fl = h5py.File(self.file_name, 'r')
        #------------------------#    
        for c, evi in enumerate(list_IDs_temp):
            dataset = fl.get('data/'+str(evi))
            if evi.split('_')[-1] == 'NO':
                pt[c] = -1
                st[c] = -2
            else:
                pt[c] = dataset.attrs['p_arrival_sample']
                st[c] = dataset.attrs['s_arrival_sample']  

            if pt[c] <=0:
                try:
                    tmp = np.array(dataset)
                except:
                    tmp = np.zeros((6000,3))
                    print(evi)
                train_x[c,:,:3]=tmp             
            else:
                train_x[c,:,:3] = np.array(dataset) 

            hp=random.randint(20,45) 
            sig_data1=train_x[c,:,:3].transpose(1,0)
            sig_data1 = self._bp_filter(sig_data1,2,1,hp,0.01)
            sig_data1 = self._normal3(sig_data1.transpose(1,0).reshape(1,6000,3))
            train_x2[c,:,:3]=sig_data1[0,:,:]    
        
            temp=(train_x[c,:,:3]).transpose(1,0)
            temp=self._bp_filter(temp,2,1,45,0.01)
            if np.max(abs(temp))>0:
                temp=temp/np.max(abs(temp))

            for ii in range(3):
                train_y[c,:,0,ii],train_x[c,:,ii]=self._inte_env(temp[ii,:],100,pt[c],index=st[c],k=c+c1)
                    
            for ii in range(3):
                train_y2[c,:,0,ii],train_x2[c,:,ii]=self._inte_env(train_x2[c,:,ii],100,pt[c],index=st[c],k=c+c1+1)

            maxn=int(st[c])
            if maxn>0 and maxn<6000:
                ln=6000-maxn                     

                train_x3[c,:ln,:]=train_x2[c,maxn:,:]
                train_y3[c,:ln,0,:]=train_y2[c,maxn:,0,:]
                
                cf=np.random.randint(0,2)
                if cf>0:
                    train_x3[c,ln:,:]=train_x2[c,:maxn,:] 
                    train_y3[c,ln:,0,:]=train_y2[c,:maxn,0,:]
                else:
                    train_x3[c,ln:,:]=train_x2[c,-maxn:,:] 
                    train_y3[c,ln:,0,:]=train_y2[c,-maxn:,0,:]
                cf=np.random.randint(0,3)
                if cf==0:
                    train_x3[c,:,:]=train_x3[c,::-1,:] 
                    train_y3[c,:,0,:]=train_y3[c,::-1,0,:]
 
        fl.close()
            
        train_x=train_x.reshape(batch_size,6000,1,3)
        train_x2=train_x2.reshape(batch_size,6000,1,3)
        train_x3=train_x3.reshape(batch_size,6000,1,3)
        
        return np.vstack([train_x,train_x2,train_x3]), np.vstack([train_y,train_y2,train_y3])     
        # return np.vstack([train_x,train_x2]), np.vstack([train_y,train_y2]) 
    
 # In[]
class DataGenerator_det(Sequence):
    
    def __init__(self, list_IDs,file_name,batch_size=128,db_no=[5,20],classfl=6,shuffle=True,qc=False,nos=False):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.file_name = file_name
        self.list_IDs = list_IDs
        self.indexes=np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.db_no=db_no
        self.classfl=classfl
        self.qc=qc
        self.nos=nos
        # os.environ["CUDA_VISIBLE_DEVICES"] = ['']
        
    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.list_IDs) // self.batch_size
    

    def __getitem__(self, index):
        # get batch data index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]        
        
        # read batch data
        if self.qc:
            X, Y, Z= self._read_data(list_IDs_temp,indexes[0],qc=self.qc) 
            return ({'wave_input': X}, {'det':Y },{'wave':Z}) 
        else:
            X, Y= self._read_data(list_IDs_temp,indexes[0],qc=self.qc)     
            return ({'wave_input': X}, {'DET':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData
        
    def _gen_label(self,itp,its):
        target = np.zeros((6000,1,2))
        if itp>target.shape[0] or itp<0:
            pass  
        target[int(itp):int(its), 0, 0]=1
    
        ite=int(itp+3*(its-itp))
        if ite>6000:
            ite=6000  
        target[int(its):int(ite), 0, 1]=1
        return target     
    
    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2

    def _add_zeros(self,data1,y1=None,k=0):
        np.random.seed(k) 
        n3=np.random.randint(0,4)
        if n3<1:
                n1=np.random.randint(500,2000)
                n4=np.random.randint(100,6000-n1-100)
                data1[n4:n4+n1]=0.0
                if not y1 is None:
                    y1[n4:n4+n1]=(np.mean(y1[:n4])+np.mean(y1[n4+n1:]))/2
        if not y1 is None:            
            return data1,y1  
        else:
            return data1


    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1  
    def _read_data(self, list_IDs_temp,c1,qc):
        """Read a batch data.
        ---
        # Arguments
            batch_files: the file of batch data.

        # Returns
            images: (batch_size, (image_shape)).
            labels: (batch_size, (label_shape)). """
        batch_size=self.batch_size
        classfl=self.classfl
        # env_model=self.env_model
        
        train_x=np.zeros((batch_size,6000,3))
        train_x2=np.zeros((batch_size,6000,1,3))
        train_x3=np.zeros((batch_size,6000,1,3))
        train_x4=np.zeros((batch_size,6000,1,3))
        
        pt=np.zeros((batch_size,))  
        st=np.zeros((batch_size,))
        dis=np.zeros((batch_size,))
        train_y2=np.zeros((batch_size,)) 
        train_y3=np.zeros((batch_size,)) 
        
        fl = h5py.File(self.file_name, 'r')
        
        #------------------------#    
        for c, evi in enumerate(list_IDs_temp):

            dataset = fl.get('data/'+str(evi)) 
            try:
                train_x[c,:,:3] = np.array(dataset)
            except:
                continue
            
            if evi.split('_')[-1] == 'NO':
                pt[c] = -1
                st[c] = -2
                train_y2[c]=0 
                train_x[c,:,:3] = np.array(dataset)
                
            else:
                try:
                    
                    pt[c] = dataset.attrs['p_arrival_sample']
                    st[c] = dataset.attrs['s_arrival_sample']
                    dis[c]= dataset.attrs['source_distance_km']
                except:
                    continue

                if classfl==2:
                    train_y2[c]=1
                else:
                    if dis[c]<30:train_y2[c]=1
                    elif dis[c]<60:train_y2[c]=2
                    elif dis[c]<90:train_y2[c]=3
                    elif dis[c]<120:train_y2[c]=4
                    else:train_y2[c]=5 

                train_x[c,:,:3] = np.array(dataset)
                
            sig_data2=self._normal3(train_x[c,:,:3].reshape(1,6000,3))
            sig_data2=sig_data2[0,:,:].transpose(1,0)
            for ii in range(3):
                sig_data2[ii,:]=self._taper(sig_data2[ii,:],1,100)                     
            sig_data2 =self._bp_filter(sig_data2,2,1,45,0.01) 
            train_x[c,:,:3]=sig_data2[:,:].transpose(1,0)                
            
            hp=random.randint(20,45) 
            hp1=random.randint(1,5)
            sig_data1 = np.zeros((1,6000,3))
            sig_data1 =self._normal3(train_x[c,:,:3].reshape(1,6000,3))
            sig_data1=sig_data1[0,:,:].transpose(1,0)
                   
            for ii in range(3):
                sig_data1[ii,:]=self._taper(sig_data1[ii,:],1,100)                     
            sig_data1 =self._bp_filter(sig_data1,2,hp1,hp,0.01) 
            train_x2[c,:,0,:3]=sig_data1[:,:].transpose(1,0)

            ccc=np.random.randint(0,4)
            
            if ccc==0:
                train_x3[c,:,0,:3]=train_x2[c,:,0,:3]
            elif ccc==1:
                train_x3[c,:,0,:3]=train_x[c,:,:3]            
            elif ccc==2:
                train_x3[c,:,0,:3]=train_x2[c,:,0,:3]
            else:
                train_x3[c,:,0,:3]=train_x[c,:,:3]
                ## time shifted
            if ccc==0 or ccc==1  and st[c]>100 and st[c]<5800:
                
                cc=np.random.randint(0,int((5900-st[c])/100))
                
                ed=6000-cc*100
                
                tmp=np.zeros((6000,3))
                tmp[cc*100:,:]=train_x3[c,:ed,0,:]
                tmp[:cc*100,:]=train_x3[c,ed:,0,:3] 
                train_x3[c,:,0,:]=tmp
                
     
            if ccc==2 or ccc==3:

                maxn=np.argmax(abs(train_x2[c,:,0,2]))
                cc=np.random.randint(0,3)
                cf=np.random.randint(1,5)
                cf1=np.random.randint(1,5)
                ed=int(maxn+cf*100)
                if ed>5800:
                    ed=5800
                star=int(maxn-cf1*100)
                if star<200:
                    star=200
                if star<ed:    
                    try:    
                        if cc==0:
                            train_x3[c,:ed-star,0,:]=train_x3[c,star:ed,0,:3]
                        if cc==1:
                            train_x3[c,star-ed:,0,:]=train_x3[c,star:ed,0,:3]
                        if cc==2:
                            train_x3[c,star-ed:,0,:]=train_x3[c,star:ed,0,:3]
                            ed=int(maxn+cf1*100)
                            if ed>5800:
                                ed=5800
                            star=int(maxn-cf*100)
                            if star<200:
                                star=200
                            if star<ed:    
                                train_x3[c,:ed-star,0,:]=train_x3[c,star:ed,0,:3]
                    except:
                        print(ed-star,star,ed)
            maxn=np.argmax(abs(train_x2[c,:,0,2])) 
            if maxn<st[c]:
                maxn=int(st[c])
            if maxn<1:
                maxn=1
                
            if ccc>1:
                cf=np.random.randint(0,5)
                maxn=int(maxn+cf*100)
                if maxn>5800:
                    maxn=5800
                if maxn<200:
                    maxn=200
                train_x4[c,:6000-maxn,0,:3]=train_x2[c,maxn:,0,:3]  
                train_x4[c,6000-maxn-100:,0,:3]=train_x2[c,-maxn-100:,0,:3] 
            else:   
                
                train_x4[c,:6000-maxn,0,:3]=train_x2[c,maxn:,0,:3]  
                train_x4[c,6000-maxn:,0,:3]=train_x2[c,-maxn:,0,:3] 
            
                        
        train_x[:,:,:3]=self._normal3(train_x[:,:,:3])
        train_x2[:,:,0,:3]=self._normal3(train_x2[:,:,0,:3])
        train_x3[:,:,0,:3]=self._normal3(train_x3[:,:,0,:3])
        train_x4[:,:,0,:3]=self._normal3(train_x4[:,:,0,:3])
        x=train_x.reshape(batch_size,6000,1,3)

        if classfl==2:
            train_y2=to_categorical(train_y2,2)
        else:   
            train_y2=to_categorical(train_y2,6)
            train_y3=to_categorical(train_y3,6)
        fl.close()
        if qc:
            return np.vstack([x,train_x2]), np.vstack([train_y2,train_y2]), np.vstack([x,train_x2]) 
        else: 
            if self.nos:
                return x,train_y2
            else:
                return np.vstack([x,train_x2,train_x3,train_x4]), np.vstack([train_y2,train_y2,train_y2,train_y3])              

class DataGenerator_test(Sequence):

    def __init__(self, list_IDs,file_name,batch_size=256,shuffle=False,add_db=0,bpfg=False,bp=45):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.file_name = file_name
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.bpfg = bpfg
        self.bp = bp
        self.add_db=add_db

    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
            
        # get batch data index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]        
        
        # read batch data
        X = self._read_data(list_IDs_temp,indexes[0])

        return ({'wave_input': X}) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise.T
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    # def _normal3(self,data):  
    #     data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
    #     for i in range(data.shape[0]):
    #         data1=data[i,:,:]
    #         x_max=np.max(abs(data1))
    #         if x_max!=0.0:
    #             data2[i,:,:]=data1/x_max 
    #     return data2
    
    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2
    
    
    def _normal3_3(self,data):
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            data1=data1-np.mean(data1,axis=0)
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0] = 1
            data2[i,:,:]=data1/x_max
        return data2
   
    def _gen_label(self,itp,its):
        target = np.zeros((6000,1,2))
        if itp>target.shape[0] or itp<0:
            pass  
        target[int(itp):int(its), 0, 0]=1
    
        ite=int(itp+3*(its-itp))
        if ite>6000:
            ite=6000  
        target[int(its):int(ite), 0, 1]=1
        return target    
    
    def _add_spike(self,data1,y1,k):
        np.random.seed(k)
        n1=np.random.randint(0,6000)
        n2=np.random.randint(0,6000)
        f1=np.min([n1,n2])
        f2=np.max([n1,n2])
        n3=np.random.randint(1,13)
        if n3<7:
            if f2!=f1:
                no1=np.load('/data2/zhangj2/ENVDET_MASTER_38/data/special_noise1.npz')['data']
                if np.max(abs(no1[f1:f2,2]))>0.5: 
                    n4=np.random.randint(0,6000-f2+f1)
                    nosie=np.zeros(len(data1),)
                    nosie[n4:n4+f2-f1]=no1[f1:f2,2]
                    data1=data1*n3/10+nosie
                    y1=y1*n3/10
        return data1,y1
    
    
    def _add_zeros(self,data1,y1,k):
        np.random.seed(k) 
        n3=np.random.randint(0,4)
        if n3<1:
                n1=np.random.randint(500,2000)
                n4=np.random.randint(100,6000-n1-100)
                data1[n4:n4+n1]=0.0
                y1[n4:n4+n1]=(np.mean(y1[:n4])+np.mean(y1[n4+n1:]))/2

        return data1,y1       
    

    def _inte_env(self,data1,d,tp1,fl=0,index=0,k=0):
        
        font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
        
        ln=len(data1)
        data2=data1.reshape(int(ln/d),d)
        data2=np.max(data2,axis=1)
    
        x=np.linspace(0,ln,int(ln/d))
        y=data2
        f=interpolate.interp1d(x,abs(y),kind='cubic')
        x1=np.linspace(0,ln,ln)
        y1=f(x1)
        
        if tp1>0:
            wed=1500
            y1[0:int(tp1)]=0
    
            if index==0:
                index=int(np.argmax(y1))
            else:
                index=int(index)
                wed=int(index-tp1)*2
                
            ed_index=index+wed
                
            if ed_index>ln:
                ed_index=6000
                
            weg=np.ones((ln,))
            weg[index:]=0
            weg[index:ed_index]=-1/wed*(np.arange(index,ed_index)-index)+1
            y1=y1*weg
        else:
            
            data1,y1=self._add_spike(data1,y1,k)    
            data1,y1=self._add_zeros(data1,y1,k)
        if fl==1:
           
            figure, ax = plt.subplots(figsize=(6,4) )  
            t=np.arange(6000)*0.01
            plt.plot(t,data1)
            plt.plot(x*0.01,y)
            plt.plot(t,y1)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            plt.xlabel('Time (s)',font2) 
            plt.ylabel('Amplitude',font2)
            plt.show()
       
        return y1,data1
        
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1      
    
    def _read_data(self, list_IDs_temp,c1):
        """Read a batch data.
        ---
        # Arguments
            batch_files: the file of batch data.

        # Returns
            images: (batch_size, (image_shape)).
            labels: (batch_size, (label_shape)). """
        batch_size=self.batch_size
        train_x=np.zeros((batch_size,6000,3))

        fl = h5py.File(self.file_name, 'r')
        #------------------------#    
        for c, evi in enumerate(list_IDs_temp):
            try:
                dataset = fl.get('data/'+str(evi))
                data = np.array(dataset)[:6000,:]  
    
                if self.bpfg:
                    if self.add_db:
                        try:
                            tp=dataset.attrs['p_arrival_sample']
                            ts=dataset.attrs['s_arrival_sample']
                            sig=data[tp:ts,:]
                        except:
                            sig=data[:500,:]
                        add_no=self._add_noise(sig,self.add_db,c)
                        data = data+add_no  
                        
                    for ii in range(3):
                        data[:,ii]=data[:,ii]-np.mean(data[:,ii]) # ZJ after discussion
                        data[:,ii]=self._taper(data[:,ii],1,100)  # before 200
                    train_x[c,:,:]=self._bp_filter(data.transpose(1,0),2,1,self.bp,0.01).transpose(1,0)
                else:
                    train_x[c,:,:] = np.array(dataset)
                train_x[c,:,:]=train_x[c,:,:]-np.mean(train_x[c,:,:],axis=0)
            except:
                print(evi)
        # train_x=self._normal3(train_x)
        
        train_x=self._normal3(train_x)
        train_x=train_x.reshape(batch_size,6000,1,3)
            
        return  train_x
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                          save_path=None,name_m='CM'):
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
    }
    plt.rc('font', family='Times New Roman')
    figure, ax = plt.subplots(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,font2)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],fontsize=12, horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]   
    
    plt.ylabel('True label',font2)
    plt.xlabel('Predicted label',font2)
    if not save_path is None:
        plt.savefig(save_path+'CM_%s.png'%name_m,dpi=600)      
    plt.show()  
    
def plot_ENVDET(test_data,test_label,pred,pred1,c=0,save_path=None,k=0):   
    data = test_data[256*k+c,:,:,:]
    lab_data = test_label[256*k+c,:,:,:]
    env_data = pred[256*k+c,:,:,:]
    det = pred1[256*k+c,:]   
    label = np.argmax(det)
    pro = det[label]
        
    font2 = {'family' : 'Times New Roman','weight' : 'bold','size'   : 18,}
    font1 = {'family' : 'Times New Roman','weight' : 'bold','size'   : 15,}
    figure, ax = plt.subplots(figsize=(8,8))
    plt.subplot(3,1,1)
    plt.plot(data[:,0,0],'k',label='Waveform')
    plt.plot(lab_data[:,0,0],'b',label='Labeled',linestyle='--')
    plt.plot(env_data[:,0,0],'r',label='Predicted',linestyle='--')
    plt.xlim([0,6000])
    plt.ylim([-1,1])
    plt.tick_params(labelsize=18)
    plt.xticks([])
    plt.text(500, 1.1, 'Predicted:',family='Times New Roman',fontsize=18)
    plt.text(1500, 1.1, '%d'%label,family='Times New Roman',fontsize=18)
    plt.text(3000, 1.1, 'Probability:',family='Times New Roman',fontsize=18)
    plt.text(4200, 1.1, '%.2f'%pro,family='Times New Roman',fontsize=18)
    plt.text(6100, 0.6, 'E',family='Times New Roman',fontsize=18)
    plt.legend(prop=font1,ncol=3)
    
    plt.subplot(3,1,2)
    plt.plot(data[:,0,1],'k',label='Waveform')
    # plt.plot(data[:,1],'k',label='Waveform')
    plt.plot(lab_data[:,0,1],'b',label='Labeled',linestyle='--')
    plt.plot(env_data[:,0,1],'r',label='Predicted',linestyle='--')
    plt.xlim([0,6000])
    plt.ylim([-1,1])
    plt.tick_params(labelsize=18)
    plt.xticks([])
    plt.text(6100, 0.6, 'N',family='Times New Roman',fontsize=18)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    plt.ylabel('Amplitude',font2)
    # [label.set_fontname('Times New Roman') for label in labels]
    # plt.text(5000, 0.5, 'SNR: %.f dB'%snr[1])
#    plt.legend(ncol=2)
    
    plt.subplot(3,1,3)
    plt.plot(data[:,0,2],'k',label='Waveform')
    # plt.plot(data[:,2],'k',label='Waveform')
    plt.plot(lab_data[:,0,2],'b',label='Labeled',linestyle='--')
    plt.plot(env_data[:,0,2],'r',label='Predicted',linestyle='--')
    plt.xlim([0,6000])
    plt.ylim([-1,1])
    plt.xticks([])
    plt.tick_params(labelsize=18)
    plt.text(6100, 0.6, 'Z',family='Times New Roman',fontsize=18)
    plt.xlabel('Samples',font2)

    if save_path:    
        plt.savefig(save_path+'ENV_EQT_%d_%d'%(k,c),dpi=600)
    plt.show()     

class DataGenerator_ENVDET(Sequence):

    def __init__(self, list_IDs,file_name,batch_size=256,shuffle=False,bp=45):
        self.batch_size = batch_size
        self.file_name = file_name
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.bp = bp
        
    def __len__(self):
        """return: steps num of one epoch. """
        
        return math.ceil(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
            
        # get batch data index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]        
        
        # read batch data
        X = self._read_data(list_IDs_temp,indexes[0])

        return ({'wave_input': X}) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2
    
    
    def _normal3_3(self,data):
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            data1=data1-np.mean(data1,axis=0)
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0] = 1
            data2[i,:,:]=data1/x_max
        return data2
      
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1      
    
    def _read_data(self, list_IDs_temp,c1):

        batch_size=len(list_IDs_temp)
        train_x=np.zeros((batch_size,6000,3))

        fl = h5py.File(self.file_name, 'r')
        #------------------------#    
        for c, evi in enumerate(list_IDs_temp):
            try:
                dataset = fl.get('data/'+str(evi))
                data = np.array(dataset)[:6000,:]  
                for ii in range(3):
                    data[:,ii]=data[:,ii]-np.mean(data[:,ii]) 
                    data[:,ii]=self._taper(data[:,ii],1,100)  
                train_x[c,:,:]=self._bp_filter(data.transpose(1,0),2,1,self.bp,0.01).transpose(1,0)
                train_x[c,:,:]=train_x[c,:,:]-np.mean(train_x[c,:,:],axis=0)
            except:
                print(evi)
        
        train_x=self._normal3(train_x)
        train_x=train_x.reshape(batch_size,6000,1,3)
            
        return  train_x    
# In[]
