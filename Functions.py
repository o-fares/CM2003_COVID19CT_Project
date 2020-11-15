#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import random 
import re
import cv2
import glob

from random import shuffle
from skimage.io import imread
from skimage.transform import resize

from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization,SpatialDropout2D,Conv2DTranspose,concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.layers import Conv2D,AveragePooling2D,add, GlobalAveragePooling2D, ReLU

from skimage.io import imshow


# In[2]:


#function that create a list from lines in a txt file
#allows to know which image is in which set
def create_list_from_txt(filename):
    FILE = filename
 
    with open(FILE, 'r') as f:
        lines = [line.strip('\n') for line in f.readlines()]
 
    list_img = [line for line in lines]
    
    return list_img
   


# In[3]:


#add the name of image to the name of folder
#create the full name of the image to be loaded
def list_dir(set_list,set_dir):
    new_data=[]
    for x in set_list:
        new = os.path.join(set_dir,x)
        new_data.append(new)
    return new_data


# In[4]:


#tbl stands for to be loaded
def create_list_to_load( covid):
    img_path='/tf/Project'
    folder_path='/tf/Project/Data_split'
    
    if covid==True:
        set_dir=os.path.join(img_path+'/CT_COVID')
        split_dir=os.path.join(folder_path+'/COVID')
    else: 
        set_dir=os.path.join(img_path+'/CT_NonCOVID')
        split_dir=os.path.join(folder_path+'/NonCOVID')
        
    list_files=glob.glob(split_dir+'/*.txt')
    list_files.sort()
    #to ensure the order of sets as test , train and val (alphabetic order)
        
    test_list=create_list_from_txt(list_files[0])
    train_list=create_list_from_txt(list_files[1])
    val_list=create_list_from_txt(list_files[2])
    
    test_list_tbl=list_dir(test_list,set_dir)
    train_list_tbl=list_dir(train_list,set_dir)
    val_list_tbl=list_dir(val_list,set_dir)
    
    return test_list_tbl, train_list_tbl, val_list_tbl


# In[5]:


def load_data(data_list,img_w,img_h,img_ch):
    tab = np.zeros((len(data_list),img_w,img_h,img_ch),dtype='float32')
    for i in range(len(data_list)):
        Img = cv2.imread(data_list[i],0)
        Img = cv2.resize(Img,(img_w, img_h))
        Img = Img.reshape(img_w,img_h)/255
        tab[i,:,:,0]=Img
    return tab


# In[6]:


def create_list_labels(set_list,covid): 
    list_labels=[]
    for i in range(len(set_list)):
        if covid==True:
            list_labels.append(1)
        else:
            list_labels.append(0)
    return list_labels


# In[7]:


def concatenate_lists_and_labels(list1,labels1,list2,labels2):
    return list1+list2, labels1+labels2


# In[8]:


def shuffle_lists(list_img, list_labels):
    index_position = list(zip(list_img, list_labels))
    random.shuffle(index_position)
    list_img[:],list_labels[:] = zip(*index_position)
    return list_img,list_labels


# In[9]:


def precision(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + K.epsilon()) / (K.sum(y_pred_f) + K.epsilon())

def recall(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + K.epsilon()) / (K.sum(y_true_f) + K.epsilon())


# In[10]:


#labels is an array
def augmentation(image_set,labels,dictionary_augmentation,batch_size):
    
    image_datagen = ImageDataGenerator(**dictionary_augmentation)

    image_generator = image_datagen.flow(
    image_set,
    y=labels,
    batch_size=batch_size,
    shuffle=False,
    seed=1)
    
    
    return image_generator


# In[12]:


def vg_model(img_ch, img_width, img_height,base_dense,dropout=False,dr=0.2,Batch_normalization=False):
    if Batch_normalization:
        
        #blue layer
        model = Sequential()
        model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
        kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))


        #orange layer
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #purple layer
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #green layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #red layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
    else:
        #blue layer
        model = Sequential()
        model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
        kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))


        #orange layer
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #purple layer
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #green layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #red layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
    if dropout:
        #dense layer
        model.add(Flatten()) 
        model.add(Dense(64))
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(64)) 
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Dropout(dr))
        model.add(Activation('sigmoid'))
    else:
        #dense layer
        model.add(Flatten()) 
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(64)) 
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
    
    
    model.summary() 
    return model


# In[13]:


def densenet(input_shape, n_classes,f=32,dropout=False,dr=0.2):
    
    def bn_rl_cv(x,f,k=1,s=1):
        x=BatchNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(f,k,strides=s, padding="same")(x)
        return x
    
    def dense_block(x,r):
        for _ in range(r):
            y=bn_rl_cv(x,4*f)
            y=bn_rl_cv(x,f,3)
            x= concatenate([y,x])
        return x
        
    
    
    def transition_layer(x):
        x=bn_rl_cv(x,K.int_shape(x)[-1]//2)
        if dropout:
            x = Dropout(dr)(x)
        x=AveragePooling2D(2, strides=2, padding="same")(x)
        return x
    
    input=Input(input_shape)
    x=Conv2D(64,7, strides=2, padding="same")(input)
    x=MaxPooling2D(3,strides=2, padding="same")(x)
    
    for r in [6,12,32,32]:
        
        d=dense_block(x,6)
        x=transition_layer(d)
    x=GlobalAveragePooling2D()(d)
    output=Dense(n_classes,activation='sigmoid')(x)
    
    model=Model(input, output)
    return model


# In[14]:


def resnet(input_shape, n_classes):
    
    def identity_block(y,f):
        x=Conv2D(f,1)(y)
        x=BatchNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(f,3,padding='same')(x)
        x=BatchNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(4*f,1)(x)
        x=BatchNormalization()(x)
        x=add([x,y])
        x=ReLU()(x)
        
        return x
    
    def conv_block(y,f,s):
        x=Conv2D(f,1)(y)
        x=BatchNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(f,3,strides=s,padding='same')(x)
        x=BatchNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(4*f,1)(x)
        x=BatchNormalization()(x)
        
        y=Conv2D(4*f,1,strides=s)(y)
        y=BatchNormalization()(y)
        
        x=add([x,y])
        x=ReLU()(x)
        
        return x
    
    def resnet_block(x,f,r,s):
        x=conv_block(x,f,s)
        for _ in range(r-1):
            x=identity_block(x,f)
        return x
    
    input=Input(input_shape)
    x=Conv2D(64,7,strides=2,padding='same')(input)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    x=MaxPooling2D(3,strides=2, padding='same')(x)
    
    x=resnet_block(x,64,3,1)
    x=resnet_block(x,128,4,2)
    x=resnet_block(x,256,6,2)
    x=resnet_block(x,512,3,2)
    
    x=GlobalAveragePooling2D()(x)
    output=Dense(n_classes,activation='sigmoid')(x)
    
    model=Model(input,output)
    return model


# In[16]:


def Alex_model(img_ch, img_width, img_height,base_dense=8,dr=0.2):
    model = Sequential()
    model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
    kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten()) 
    model.add(Dense(128))
    model.add(Dropout(dr))
    model.add(Activation('relu'))
    
    model.add(Dense(64)) 
    model.add(Dropout(dr))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Dropout(dr))
    model.add(Activation('sigmoid'))
    model.summary() 
    return model


# In[1]:


def loss_curves_plot(model_hist):
    get_ipython().run_line_magic('matplotlib', 'inline')

    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(model_hist.history["loss"], label="loss")
    plt.plot(model_hist.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(model_hist.history["val_loss"]),
     np.min(model_hist.history["val_loss"]),
     marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    
def accuracy_curves_plot(model_hist,metrics):
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(model_hist.history[metrics], label="accuracy")
    plt.plot(model_hist.history["val_"+metrics], label="val_accuracy")
    plt.plot( np.argmax(model_hist.history["val_"+metrics]),
     np.max(model_hist.history["val_"+metrics]),
     marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Value")
    plt.legend();


# In[ ]:




