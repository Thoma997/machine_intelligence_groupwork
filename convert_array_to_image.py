# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:24:29 2018

@author: Fiona Mallett

"""
#Convert numpy array to a picture
#It combines the 2 datasets, so the name given to each image either ends in '0'
#or '1' for cancerous or non-cancerous
from PIL import Image
import os

def convert_to_image(data_x, name, data_y):
    image_root_folder = ('dataset/' + name)
    if not os.path.exists(image_root_folder):
        os.makedirs(image_root_folder)
        
    cancerous = ('dataset/' + name + '/cancerous')
    if not os.path.exists(cancerous):
        os.makedirs(cancerous)   
    
    normal = ('dataset/' + name + '/normal')
    if not os.path.exists(normal):
        os.makedirs(normal) 
        
        
    for i in range(0, data_x.shape[0]):
        img = Image.fromarray(data_x[i])
        if data_y[i] == 0:
          
            filename = '%s/%s%s.png' % (normal, i, '0')
        else :
            filename = '%s/%s%s.png' % (cancerous, i, '1')
        img.save(filename)
