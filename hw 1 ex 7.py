# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:51:55 2018

@author: Elena
"""

import matplotlib.pyplot as plt
import random 
import seaborn as sns
import numpy as np
from collections import Counter

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
    
#average over 1000 simulations 
#cycle for 1000 runs
all_cycles=[]
all_missclass_probs=[]
N=10

for i in range(1000):
    print('cycle no', i)
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
    
    #The perceptron model 
    #initial weights (2D)
    w=[0,0,0]
    data_artif=np.ones(N)
    misclassified_pts=np.ones(N)
    
    points=np.array([[data_artif[pt_no], data_x[pt_no], data_y[pt_no]] for pt_no in range(N)])
    
    cycle_counter=0
    true_frac=0
    
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
        
    all_cycles.append(cycle_counter)

    #estimate the test (for probability) set
    M=10000
    x, y=pick_points(M)
    test_points=np.array([np.ones(M), x,y])   
    test_points=np.transpose(test_points)
    test_preds=[sign(np.dot(w, point)) for point in test_points]
    test_data_labels=[]
    for point in range(M):
        line_y_compare=m*x[point]+b
        if line_y_compare>y[point]:
            test_data_labels.append(-1)
        else:
           test_data_labels.append(+1)
           
    test_clasion=[test_preds[i]==test_data_labels[i] for i in range(M)]
    c=Counter(test_clasion)
    probability=c[True]/M
    
    #final parameter
    missclassification_prob=1-probability
    all_missclass_probs.append(missclassification_prob)

        
    line_x_pla=[-1,1]
    line_y_pla=[(w[1]-w[0])/w[2], (-w[1]-w[0])/w[2]]
    
    
    
#plot the last itteration
    
f1=plt.figure(1)
plt.axis([-1,1,-1,1])
plt.plot(line_x_new, line_y_new, sns.xkcd_rgb["black"])
sns.scatterplot(x=line_x, y=line_y, hue=['line', 'line'], \
                palette=sns.color_palette([sns.xkcd_rgb["black"]]))

sns.scatterplot(data_x, data_y, hue=data_labels, \
                palette=sns.color_palette([sns.xkcd_rgb['gold'], \
                                           sns.xkcd_rgb['denim blue']]))

#plot which points were misclassified   
#get line parameters y=mx+c
line_x_pla=[-1,1]
line_y_pla=[(w[1]-w[0])/w[2], (-w[1]-w[0])/w[2]]


f2=plt.figure(2)
plt.axis([-1,1,-1,1])

plt.plot(line_x_new, line_y_new, sns.xkcd_rgb["black"])
plt.plot(line_x_pla, line_y_pla, sns.xkcd_rgb["purple"])

sns.scatterplot(data_x, data_y, hue=classification, \
        palette=sns.color_palette([sns.xkcd_rgb['medium green']\
                                   ,sns.xkcd_rgb['pale red']], n_colors=len(set(classification))))
plt.legend(title='correct prediction?')
plt.title('classification of points')
#print('fraction of correct classifications {}'.format(true_frac))
  
f3=plt.figure(3)
sns.lineplot(np.arange(cycle_counter),performance_history)
plt.xlabel('cycle no')
plt.ylabel('performance')
plt.title('fraction of correct predictions over iterrations')

f4=plt.figure(4)
plt.axis([-1,1,-1,1])

plt.plot(line_x_new, line_y_new, sns.xkcd_rgb["black"])
plt.plot(line_x_pla, line_y_pla, sns.xkcd_rgb["purple"])

sns.scatterplot(x, y, hue=test_clasion, \
        palette=sns.color_palette([sns.xkcd_rgb['medium green']\
                                   ,sns.xkcd_rgb['pale red']], n_colors=len(set(test_clasion))))
plt.legend(title='correct prediction?')
plt.title('random points to estimate classification error')
#plt.show()

          
    
print('Random points {}, average cycles {}, average disagreement \
      {}'.format(N, np.average(all_cycles), np.average(all_missclass_probs)))
    
    
plt.show() 
    
    
