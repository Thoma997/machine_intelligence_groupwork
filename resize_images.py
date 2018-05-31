#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 01:53:07 2018

@author: matthiasboeker
"""

def resize_images(image_data):
    
    small_t = np.zeros((87,50),dtype= np.uint8)
    big_t = np.zeros((224,87),dtype= np.uint8)
    size = image_data.shape
    triple = np.empty((224,224,3))
    save_i = np.empty((5547,224,224,3))
    
    for i in range(0,size[0]-1):
        for k in range(0,size[3]-1):
            inbe = np.concatenate((small_t,image_data[i,:,:,k], small_t), axis=0)
            single = np.concatenate((big_t,inbe, big_t), axis=1)
            triple[:,:,k] = single  
            
        save_i[i,:,:,:] = triple 
    return save_i