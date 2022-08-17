#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 11:39:15 2022

@author: zhangj2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:02:36 2021

@author: zhangj
"""

import keras
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten,Embedding, LSTM,GRU,Bidirectional
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D,TimeDistributed 
from keras.layers import Cropping1D,Cropping2D,ZeroPadding1D,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda,concatenate,add,Conv2DTranspose,Concatenate
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.layers import GlobalAveragePooling1D,GlobalMaxPooling1D,Softmax
from keras.layers import Bidirectional,Add

from keras import backend as K
# from keras.utils import plot_model
import tensorflow as tf

def Conv2d_BN1(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2d_BN2(x, nb_filter, kernel_size, strides=(4,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN1(x, filters, kernel_size, strides=(4,1), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN2(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN3(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = UpSampling2D(size=(4,1))(x) #1
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x    
    
def crop_and_concat(net):
    """
    the size(net1) <= size(net2)
    """
    net1,net2=net
    net1_shape = net1.get_shape().as_list()
    net2_shape = net2.get_shape().as_list()
    
    #  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
    offsets = [0, 0, 0, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)
    
    return tf.concat([net1, net2_resize], 3) 
    
def crop_and_cut(net):
    """
    the size(net1) <= size(net2)
    """
    net1,net2=net
    net1_shape = net1.get_shape().as_list()
    net2_shape = net2.get_shape().as_list()
    
    #  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
    offsets = [0, 0, 0, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)
    
    return net2_resize 
    
def crop_and_cut1(net):
    """
    the size(net1) <= size(net2)
    """
    net1,net2=net
    net1_shape = net1.get_shape().as_list()
    net2_shape = net2.get_shape().as_list()
    
    #  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
    offsets = [0, 0, 0]
    size = [-1, net1_shape[1], net1_shape[2]]
    net2_resize = tf.slice(net2, offsets, size)
    
    return net2_resize     
    
# In[]
def ENV_model(wave_input,num=3):
    nb_filter=8
    kernel_size=(7,1)
    depths=5
    
    inpt = Input(shape=wave_input,name='wave_input')
    # down
    convs=[None]*depths
    net = Conv2d_BN1(inpt, nb_filter, kernel_size)
    for depth in range(depths):
        filters=int(2**depth*nb_filter)
        
        net = Conv2d_BN1(net, filters, kernel_size)
        convs[depth] = net
    
        if depth < depths - 1:
            net = Conv2d_BN2(net, filters, kernel_size)
            
    # up
    net1=net
    for depth in range(depths-2,-1,-1):
        filters = int(2**(depth) * nb_filter)  
        net1 = Conv2dT_BN3(net1, filters, kernel_size)
        # skip and concat
        net1 =Lambda(crop_and_cut)([convs[depth], net1])
                
    outenv = Conv2D(num, kernel_size=(3,1),padding='same',name='env_output')(net1)
    model = Model(inpt, [outenv],name='ENV')
    return model   
  
# In[]
def DET_model(wave_input,class_n):
    # sample model to achieve classification

    kernel_size=(3,1)
    inpt = Input(shape=wave_input,name='wave_input')
    
    x = Conv2D(32, kernel_size=kernel_size)(inpt)
    x = MaxPooling2D( pool_size=(2,1),strides=(2,1)   )(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, kernel_size=kernel_size)(x)
    x = MaxPooling2D(pool_size=(2,1),strides=(2,1)   )(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)  
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, kernel_size=kernel_size)(x)
    x = MaxPooling2D(pool_size=(2,1),strides=(2,1)   )(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x) 
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, kernel_size=kernel_size)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x) 
    x = Dropout(0.25)(x)
    
    x = Conv2D(32, kernel_size=kernel_size)(x)
    x = MaxPooling2D(pool_size=(2,1),strides=(2,1)   )(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x) 
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    
    x1 = Dense(512,activation='relu')(x) #128
    x1 = Dropout(0.25)(x1)
    
    x1 = Dense(512,activation='relu')(x1) #128 #
    x1 = Dense(512,activation='tanh')(x1)
    x1 = Dropout(0.25)(x1)     
    
    out1 = Dense(class_n,activation='softmax',name='det')(x1)   

    model = Model(inpt, out1,name='DET')
    
    return model

# In[] balance loss 
from itertools import product
from functools import partial, update_wrapper

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = weights.shape[1]#len(weights[0,:])

    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)#returns maximum value along an axis in a tensor
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] *y_pred_max_mat[:, c_p]*y_true[:, c_t])
    # print(final_mask)
    return K.categorical_crossentropy(y_true,y_pred) * final_mask

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


# In[] build model
def bmodel(wave_input,env_model_name,cl,wts):
    z = Input(wave_input,name='wave_input')
    model=ENV_model(wave_input)
    model.trainable = False
    model.load_weights(env_model_name)
    model1=DET_model(wave_input,cl) 
    x=model(z)
    y=model1(x)
    model2=Model(z,y)
    #========#
    # model2.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    # =======#
    ncce = wrapped_partial(w_categorical_crossentropy, weights=wts)
    model2.compile(loss=ncce,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    #=======#
    return model2

# In[]
def my_reshape(x,a,b):
    return K.reshape(x,(-1,a,b)) 

def DET_master2(wave_input,class_n):
    # sample model to achieve classification

    kernel_size=(3,1)
    inpt = Input(shape=wave_input,name='wave_input')
    
    x = Conv2D(32, kernel_size=kernel_size)(inpt)
    x = MaxPooling2D( pool_size=(2,1),strides=(2,1)   )(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(64, kernel_size=kernel_size)(x)
    x = MaxPooling2D(pool_size=(2,1),strides=(2,1)   )(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)  

    x = Conv2D(128, kernel_size=kernel_size)(x)
    x = MaxPooling2D(pool_size=(2,1),strides=(2,1)   )(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x) 

    x = Conv2D(256, kernel_size=kernel_size)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x) 
    x = Lambda(my_reshape,arguments={'a':746,'b':256})(x)
    x = Bidirectional(LSTM(4,return_sequences=True))(x)
    x = TimeDistributed(Dense(8, activation='relu'))(x)    
    
    x = Flatten()(x)
    
    x1 = Dense(512,activation='relu')(x) #128
    x1 = BatchNormalization()(x1)
    
    x1 = Dense(512,activation='relu')(x1) #128
    x1 = BatchNormalization()(x1)
    
    out1 = Dense(class_n,activation='softmax',name='det')(x1)   

    model = Model(inpt, out1,name='DET')
    
    return model




 