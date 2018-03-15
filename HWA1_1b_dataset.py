#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:45:20 2018

@author: martinthoma
"""

#import libs
import numpy as np


# Setting up Dataset for HWA 1b (costum spirals)
i = np.arange(0 , 97 , 1, dtype = int)
rho = [element / 16 * np.pi for element in i]
r = [6.5 * (104 - element) / 104 for element in i]

x_s1 = []
a = 0
for a in range(len(i)):
    x_s1.append(r[a] * np.cos(rho[a] + np.pi/2))

y_s1 = []
for a in range(len(i)):
    y_s1.append(r[a] * np.sin(rho[a] + np.pi/2))
    
x_s2 = []
a = 0
for a in range(len(i)):
    x_s2.append(r[a] * np.cos(rho[a] + np.pi))

y_s2 = []
for a in range(len(i)):
    y_s2.append(r[a] * np.sin(rho[a] + np.pi))
    
x_s3 = []
a = 0
for a in range(len(i)):
    x_s3.append(r[a] * np.cos(rho[a]))

y_s3 = []
for a in range(len(i)):
    y_s3.append(r[a] * np.sin(rho[a]))




#imports for plotting
import matplotlib.pyplot as plt

#plotting spirals
plt.title('Two-Spirals-Problem')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.scatter(x_s1 , y_s1 , c = 'red')
plt.scatter(x_s2 , y_s2 , c = 'blue')
plt.scatter(x_s3 , y_s3 , c = 'green')
plt.show()