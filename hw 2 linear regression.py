# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:40:46 2018

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
    
#cycle for 1000 runs
E_ins=[]
steps_converge=[]
E_outs=[]
gs=[]
N=10
repeats=1000

for run in range(repeats):
    
    #pick a line:
    line_x, line_y=pick_points(2)
    
    #get line parameters y=mx+c
    m=(line_y[1]-line_y[0])/(line_x[1]-line_x[0])
    b=line_y[0]-m*line_x[0]
    
    line_x_new=[-1,1]
    line_y_new=[m*-1+b, m+b]
    data_x, data_y=pick_points(N)
    
    data_labels=[]
    #determine label:
    for point in range(len(data_x)):
        line_y_compare=m*data_x[point]+b
        if line_y_compare>data_y[point]:
            data_labels.append(-1)
        else:
            data_labels.append(+1)
            
        
    #linear regression
    
    X=np.transpose(np.matrix([np.ones(N), data_x, data_y]))
    Y=np.transpose(np.matrix([data_labels]))
    
    w=np.linalg.inv(np.transpose(X)*X)*np.transpose(X)*Y
    w=np.array(w).reshape(3)
    gs.append(w)
    
    #classify predictions
    points=np.array([[np.ones(N)[pt_no], data_x[pt_no], data_y[pt_no]] for pt_no in range(N)])
    all_preds=[sign(np.dot(w, point)) for point in points]
    classification=[all_preds[i]==data_labels[i] for i in range(N)]
            
    c=Counter(classification)
    e_in=c[False]/N
    E_ins.append(e_in)
    
    
    #generate 1000 points to get out of sample error 
    N_new=1000
    
    data_x_new, data_y_new=pick_points(N_new)
    onew=np.ones(N_new)
    points_new=np.array([[onew[pt_no], data_x_new[pt_no], data_y_new[pt_no]] for pt_no in range(N_new)])

    line_x_lr=[-1,1]
    line_y_lr=[(w[1]-w[0])/w[2], (-w[1]-w[0])/w[2]]

    data_labels_new=[]
    for point in range(N_new):
        line_y_compare=m*data_x_new[point]+b
        if line_y_compare>data_y_new[point]:
            data_labels_new.append(-1)
        else:
            data_labels_new.append(+1)

    preds_new=[sign(np.dot(w, point)) for point in points_new]
    classification_new=[preds_new[i]==data_labels_new[i] for i in range(N_new)]
    
    c_new=Counter(classification_new)
    e_out=c_new[False]/N_new
    E_outs.append(e_out)
    
    #take the LR weights for PLA
    
    cycle_counter=0
    true_frac=c[True]/N
    performance_history=[]
    
    while true_frac<1:
        
        pt_no=random.randint(0,N-1)
        
        point=points[pt_no]
        h=sign(np.dot(w,point))
        
        if h!=data_labels[pt_no]: 
            cycle_counter+=1
            #update the weights if misclassified
            change=np.dot(point, data_labels[pt_no])
            w=w+change
        
            all_preds=[sign(np.dot(w, point)) for point in points]
            classification=[all_preds[i]==data_labels[i] for i in range(N)]
            c=Counter(classification)
            true_frac=c[True]/N
            performance_history.append(true_frac)
            
    line_x_pla=[-1,1]
    line_y_pla=[(w[1]-w[0])/w[2], (-w[1]-w[0])/w[2]]

    
    steps_converge.append(cycle_counter)
    
    

print('Average Ein=', np.average(E_ins))
print('Average Eout=', np.average(E_outs))
print('Steps to converge=', np.average(steps_converge))

palette={True: sns.xkcd_rgb['medium green'], False:sns.xkcd_rgb['pale red']}

#visualise one run
f1=plt.figure(1)
plt.axis([-1,1,-1,1])
plt.plot(line_x_new, line_y_new, sns.xkcd_rgb["black"])
sns.scatterplot(x=line_x, y=line_y, hue=['line', 'line'], \
                palette=sns.color_palette([sns.xkcd_rgb["black"]]))

sns.scatterplot(data_x, data_y, hue=data_labels, \
                palette=sns.color_palette([sns.xkcd_rgb['gold'], \
                                           sns.xkcd_rgb['denim blue']]))
 
f2=plt.figure(2)
plt.axis([-1,1,-1,1])

plt.plot(line_x_new, line_y_new, sns.xkcd_rgb["black"], label='true f-ion')
plt.plot(line_x_lr, line_y_lr, sns.xkcd_rgb["purple"], label='hypothesis')

sns.scatterplot(data_x, data_y, hue=classification, palette=palette)
plt.legend(title='correct prediction?')
plt.title('classification of points in sample')

 
f3=plt.figure(3)
plt.axis([-1,1,-1,1])

plt.plot(line_x_new, line_y_new, sns.xkcd_rgb["black"], label='true f-ion')
plt.plot(line_x_lr, line_y_lr, sns.xkcd_rgb["purple"], label='hypothesis')

sns.scatterplot(data_x_new, data_y_new, hue=classification_new, palette=palette)
plt.legend(title='correct prediction?')
plt.title('classification of points out of sample')


f4=plt.figure(4)
plt.axis([-1,1,-1,1])

plt.plot(line_x_new, line_y_new, sns.xkcd_rgb["black"], label='true f-ion')
plt.plot(line_x_pla, line_y_pla, sns.xkcd_rgb["purple"], label='hypothesis')

sns.scatterplot(data_x_new, data_y_new, hue=classification_new, palette=palette)
plt.legend(title='correct prediction?')
plt.title('classification of points by perceptron la')

