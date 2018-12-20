# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:49:03 2018

@author: Elena
"""

import numpy as np 
import random 
import matplotlib.pyplot as plt
import seaborn as sns

v1_dist=[]
vrnd_dist=[]
vmin_dist=[]

print_step=400
#simulation run 100,000 times
for k in range(100000):
    if k%print_step==0:
        print(k)
    coins=[]
    #1 corresponds to heads, 0 to tails
    states=[0,1]
    
    ncoins=1000
    #flip each of 1000 coin 10 times
    for j in range(ncoins):
        flips=[]
        for i in range(10):
            flip=random.choice(states)
            flips.append(flip)
        coins.append(flips)
        
    frequency=np.sum(coins, axis=1)/10
    min_val=min(frequency)
    
    rand_int=random.randint(0,999)
    rnd_val=frequency[rand_int]

    v1_dist.append(frequency[0])
    vrnd_dist.append(rnd_val)
    vmin_dist.append(min_val)
    
f1=plt.figure(1)
sns.distplot(v1_dist, kde=False, bins=11)
plt.title('\'Heads\' frequency distribution for the first coin')

f2=plt.figure(2)
sns.distplot(vrnd_dist,kde=False, bins=11)
plt.title('\'Heads\' frequency distribution for a random coin')

f3=plt.figure(3)
sns.distplot(vmin_dist, kde=False, bins=2)
plt.title('\'Heads\' frequency distribution for coin with the lowest frequency')


plt.show()
print('average for c1', np.average(v1_dist))



