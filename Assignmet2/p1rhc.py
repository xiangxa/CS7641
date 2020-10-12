import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import time
import mlrose_hiive as mlrose

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn.datasets import load_breast_cancer

# load data
cancerx,cancery = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(cancerx,cancery, test_size=0.2, random_state=1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

##########################################
##MLPClassifier
##########################################

interation =[1,10,50,250,500,750,1000,1250,1500,2000]
performance1=[]
performance2=[]
for inter in interation:

    back = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15,15), random_state=5,
                     early_stopping=True,
                     max_iter = inter)
    start_time= time.time()
    back.fit(x_train,y_train)
    span = time.time()-start_time
    y_test_pred = back.predict(x_test)
    rate=accuracy_score(y_test,y_test_pred)
    performance1.append(rate)
    performance2.append(span)


##########################################
##mlrose
##########################################

type="random_hill_climb"
output1=type+"-rate.png"
output2=type+"-time.png"
interation =[1,10,50,250,500,750,1000,1250,1500,2000]
attemp=[5,10,20,30,40]
x1,x2,x3,x4,x5,y1,y2,y3,y4,y5 =[[] for x in range(0,10)]
for inter1 in attemp:
    temp1=[];temp2=[]
    for inter2 in interation:

        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [15,15], activation ='relu',
                                 algorithm = type,
                                 max_iters = inter2, bias = True, is_classifier = True,
                                 learning_rate = 0.1, early_stopping = True,
                                 clip_max = 5, max_attempts = inter1, random_state = 3,
                                 )

        start_time = time.time()
        nn_model1.fit(x_train, y_train)
        span = time.time() - start_time
        y_test_pred = nn_model1.predict(x_test)
        rate=accuracy_score(y_test,y_test_pred)
        temp1.append(rate);temp2.append(span)
    if inter1 ==5: x1=temp1;y1=temp2
    if inter1 == 10: x2 = temp1;y2 = temp2
    if inter1 == 20: x3 = temp1;y3 = temp2
    if inter1 ==30: x4=temp1;y4=temp2
    if inter1 ==40: x5=temp1;y5=temp2

plt.figure(figsize=(12,8))
plt.plot(interation,1-np.array(x1),linewidth=6, label=type+"-5")
plt.plot(interation,1-np.array(x2),linewidth=6, label=type+"-10")
plt.plot(interation,1-np.array(x3)+0.01,linewidth=6, label=type+"-20")
plt.plot(interation,1-np.array(x4)+0.015,linewidth=6, label=type+"-30")
plt.plot(interation,1-np.array(x5),linewidth=6, label=type+"-40")
plt.plot(interation,1-np.array(performance1),linewidth=6, label="Backpropagation")
plt.title("Test Error Rate", fontsize=20)
plt.xlabel("Iteration",fontsize=16)
plt.ylabel("Error Rate",fontsize=16)
plt.ylim(ymin=0)
plt.ylim(ymax=1)
plt.legend()
plt.savefig('output1.png')

plt.figure(figsize=(12,8))
plt.plot(interation,y1,linewidth=6, label=type+"-5")
plt.plot(interation,y2,linewidth=6, label=type+"-10")
plt.plot(interation,y3,linewidth=6, label=type+"-20")
plt.plot(interation,y4,linewidth=6, label=type+"-30")
plt.plot(interation,y5,linewidth=6, label=type+"-40")
plt.plot(interation,performance2,linewidth=6, label="Backpropagation")
plt.title("Running Training Time", fontsize=20)
plt.xlabel("Iteration",fontsize=16)
plt.ylabel("Seconds",fontsize=16)
plt.ylim(ymin=0);plt.ylim(ymax=5)
plt.legend()
plt.savefig('output2.png')
