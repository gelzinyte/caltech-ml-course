# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:52:13 2018

@author: Elena
"""


import matplotlib.pyplot as plt
import random 
import seaborn as sns
import numpy as np
from collections import Counter
import time

#begining same as perceptron
#set up the data
#pick random point from input space X E ([-1,1];[-1,1])
def pick_points(no_points):
    all_x=[]
    all_y=[]
    for i in range(no_points):
        x=random.uniform(-1,1)
        y=random.uniform(-1,1)
        all_x.append(x)
        all_y.append(y)
    return all_x, all_y

def sign (number):
    if number<0:
        return -1
    elif number>0:
        return 1
    elif number==0:
        return 0

N=1000
runs=1000
E_ins_LR=[]
E_outs=[]
w_tilde=[]
fraction_correct=[]

for run in range(runs):

    data_x, data_y=pick_points(N)
    
    #determine label f(x)=sign(x1^2+x2^2-0.6)
    data_labels=[sign(data_x[point]**2+data_y[point]**2-0.6) for point in range(N)]
    
    #generate noise
    incorrect=[random.randint(0,N-1) for i in range(int(N*0.1))]
    for thing in incorrect:
        data_labels[thing]*=-1
        
    onew=np.ones(N)
    points=np.array([[onew[pt_no], data_x[pt_no], data_y[pt_no]] for pt_no in range(N)])
    ''' Q8 '''
    #Linear Regression without transformation
        
    X=np.transpose(np.matrix([np.ones(N), data_x, data_y]))
    Y=np.transpose(np.matrix([data_labels]))
    
    w=np.linalg.inv(np.transpose(X)*X)*np.transpose(X)*Y
    w=np.array(w).reshape(3)
    
    #find classification for linear regression
    pred_labels=[sign(np.dot(w,point)) for point in points]
    classification=[pred_labels[i]==data_labels[i] for i in range(len(points))]
            
    c=Counter(classification)
    e_in=c[False]/N
    E_ins_LR.append(e_in)

    '''Q9'''
    
    #Linear Regressor on non-linearly transofrmed data
    points_nonl=np.array([[points[i][0], points[i][1], points[i][2], points[i][1]*points[i][2], \
                 points[i][1]**2, points[i][2]**2] for i in range(len(points))])
    
    X=points_nonl
    w_nonl=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    w_nonl=np.array(w_nonl).reshape(6)
    
    w_tilde.append(w_nonl)
   
     
    #check agreement with given hypothesis 
    pred_labels_nonl=[sign(np.dot(w_nonl,point)) for point in points_nonl]
    #change to find the correct answer
    answ_labels=[sign(-1-0.05*points_nonl[i][0]+0.08*points_nonl[i][1]+0.13*points_nonl[i][3]\
                      +1.5*points_nonl[i][4]+1.5*points_nonl[i][5]) for i in range(N)]
    
    classification_nonl=[pred_labels_nonl[i]==answ_labels[i] for i in range(N)]
            
    c=Counter(classification_nonl)
    correct_class=c[True]/N
    fraction_correct.append(correct_class)
    
    
    '''Q10'''
    #Estimate out-of-sample error
    data_x_out, data_y_out=pick_points(N)
    points_out=np.array([[onew[pt_no], data_x_out[pt_no], data_y_out[pt_no]] for pt_no in range(N)])

    
    #determine label f(x)=sign(x1^2+x2^2-0.6)
    data_labels_out=[sign(data_x_out[point]**2+data_y_out[point]**2-0.6) for point in range(N)]
    
    #generate noise
    incorrect=[random.randint(0,N-1) for i in range(int(N*0.1))]
    for thing in incorrect:
        data_labels_out[thing]*=-1
        
    points_out=np.array([[points_out[i][0], points_out[i][1], points_out[i][2], points_out[i][1]*points_out[i][2], \
                 points_out[i][1]**2, points_out[i][2]**2] for i in range(len(points_out))])

    #find classification for linear regression
    pred_labels_out=[sign(np.dot(w_nonl,point)) for point in points_out]
    classification_out=[pred_labels_out[i]==data_labels_out[i] for i in range(N)]
            
    c=Counter(classification_out)
    e_out=c[False]/N
    E_outs.append(e_out)


    
w_tilde=np.array(w_tilde)
print('Average in-sample error for Linear Regressor: {}'.format(np.average(E_ins_LR)))
print('Average in-sample error for Linear Regressor on non-linearly transformed data:', np.average(w_tilde, axis=0))
print('Average agreement', np.average(fraction_correct))
print('Average out-of-sample error', np.average(E_outs))


palette={True: sns.xkcd_rgb['medium green'], False:sns.xkcd_rgb['pale red']}
palette_2={1: sns.xkcd_rgb['gold'], -1: sns.xkcd_rgb['denim blue']}

#visualise one run
f1=plt.figure(1)
sns.scatterplot(data_x, data_y, hue=data_labels, \
                palette=palette_2)

f2=plt.figure(2)
plt.axis([-1,1,-1,1])

sns.scatterplot(data_x, data_y, hue=classification_nonl, palette=palette)
plt.legend(title='correct prediction?')
plt.title('agreement with answer\'s hypothesis')

f3=plt.figure(3)
sns.scatterplot(points_out[:,1], points_out[:,2], hue=classification_out, palette=palette)
plt.legend(title='correct prediction?')
plt.title('out of sample points')


