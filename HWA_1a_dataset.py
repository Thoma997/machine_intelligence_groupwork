# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

""" creating the dataset for the origin two spiral task"""

#import libs
import numpy as np


# Setting up Dataset for HWA 1b (costum spirals)
i = np.arange(0 , 97 , 1, dtype = int)
rho = [element / 16 * np.pi for element in i]
r = [6.5 * (104 - element) / 104 for element in i]

x_s1 = []
a = 0
for a in range(len(i)):
    x_s1.append(r[a] * np.cos(rho[a]))

y_s1 = []
for a in range(len(i)):
    y_s1.append(r[a] * np.sin(rho[a]))
    

x_s2 = [element * (-1) for element in x_s1]
y_s2 = [element * (-1) for element in y_s1]


#imports for plotting
import matplotlib.pyplot as plt

#plotting spirals
plt.title('Two-Spirals-Problem (unmodified')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.scatter(x_s1 , y_s1 , c = 'red')
plt.scatter(x_s2 , y_s2 , c = 'blue')
plt.show()

# creating x and y variable for whole project. All x and y vars of the two spirals 
# together

x = []
for element in x_s1:
    x.append(element)
    
for element in x_s2:
    x.append(element)

y = []
for element in y_s1:
    y.append(element)
    
for element in y_s2:
    y.append(element)
