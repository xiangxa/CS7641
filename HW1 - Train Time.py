
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import time

irisX,irisY = load_iris(return_X_y=True)
cancerx,cancery = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(cancerx, cancery, test_size=0.2, random_state=1)

#Decision Tree
defaultDT = tree.DecisionTreeClassifier()

accuracy_test = []
accuracy_train = []
Train_Time = []
for i in range(1,99):
    train_size = i/100
    train_x, test_x, train_y, test_y = train_test_split(irisX,irisY , random_state=0, test_size=1 - i / 100)
    DTP = tree.DecisionTreeClassifier(min_samples_split=80)
    TT = time.time()
    DTP = DTP.fit(x_train, y_train)
    TT = time.time() - TT
    y_predp_test = DTP.predict(x_test)
    y_predp_train = DTP.predict(x_train)
    dtpaccuracy_test = accuracy_score(y_test, y_predp_test)
    dtpaccuracy_train = accuracy_score(y_train, y_predp_train)
    accuracy_test.append(dtpaccuracy_test)
    accuracy_train.append(dtpaccuracy_train)
    Train_Time.append(TT)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Test Size")
plt.ylabel("Accuracy")
plt.title("Decision Tree of Training and Testing - iris ")
plt.legend()
plt.savefig("DTAccuracyTT.png")
plt.clf()

plt.plot(Train_Time, label='Train Size')
plt.xlabel("Train Size")
plt.ylabel("Training Time")
plt.title("Decision Tree Training Time - iris ")
plt.legend()
plt.savefig("DTTT.png")
plt.clf()
######



#Boost 
accuracy_test = []
accuracy_train = []
baccuracy_test = []
baccuracy_train = []
Train_Time = []

for i in range(1,99):
    train_size = i/100
    train_x, test_x, train_y, test_y = train_test_split(irisX,irisY , random_state=0, test_size=1 - i / 100)
    DTP = tree.DecisionTreeClassifier(min_samples_split=80)
    DTP.fit(x_train, y_train)
    Boost = AdaBoostClassifier(base_estimator=DTP,n_estimators=50,random_state=0)
    TT = time.time()
    Boost.fit(x_train, y_train)
    TT = time.time() - TT
    y_pred_test = DTP.predict(x_test)
    y_pred_train = DTP.predict(x_train)
    by_pred_test = Boost.predict(x_test)
    by_pred_train = Boost.predict(x_train)
    DTaccuracy_test = accuracy_score(y_test, y_pred_test)
    DTaccuracy_train = accuracy_score(y_train, y_pred_train)
    baccuracy_test1 = accuracy_score(y_test, by_pred_test)
    baccuracy_train1 = accuracy_score(y_train, by_pred_train)
    accuracy_test.append(DTaccuracy_test)
    accuracy_train.append(DTaccuracy_train)
    baccuracy_test.append(baccuracy_test1)
    baccuracy_train.append(baccuracy_train1)
    Train_Time.append(TT)

plt.plot(accuracy_test, label='DTtest')
plt.plot(accuracy_train, label='DTtrain')
plt.plot(baccuracy_test, label='Boosttest')
plt.plot(baccuracy_train, label='Boosttrain')

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Boosting Decision Tree Training and Testing - 50 estimators Split 80- iris")
plt.legend()
plt.savefig("DDTBoostAccuracyTT.png")
plt.clf()

plt.plot(Train_Time)
plt.xlabel("Train Size")
plt.ylabel("Training Time")
plt.title("Boosting Training Time - iris ")
plt.legend()
plt.savefig("BoostTT.png")
plt.clf()
'''
accuracy_test = []
accuracy_train = []
baccuracy_test = []
baccuracy_train = []
Train_Time = []

for i in range(1,50):
    DTP = tree.DecisionTreeClassifier(min_samples_split=80)
    DTP.fit(x_train, y_train)
    Boost = AdaBoostClassifier(base_estimator=DTP,n_estimators=i,random_state=0)
    TT = time.time()
    Boost.fit(x_train, y_train)
    TT = time.time() - TT
    y_pred_test = DTP.predict(x_test)
    y_pred_train = DTP.predict(x_train)
    by_pred_test = Boost.predict(x_test)
    by_pred_train = Boost.predict(x_train)
    DTaccuracy_test = accuracy_score(y_test, y_pred_test)
    DTaccuracy_train = accuracy_score(y_train, y_pred_train)
    baccuracy_test1 = accuracy_score(y_test, by_pred_test)
    baccuracy_train1 = accuracy_score(y_train, by_pred_train)
    accuracy_test.append(DTaccuracy_test)
    accuracy_train.append(DTaccuracy_train)
    baccuracy_test.append(baccuracy_test1)
    baccuracy_train.append(baccuracy_train1)
    Train_Time.append(TT)


plt.plot(Train_Time)
plt.xlabel("number of estimators")
plt.ylabel("Training Time")
plt.title("Boosting Training Time - Cancer dataset ")
plt.legend()
plt.savefig("BoostTTestimator.png")
plt.clf()


'''

#KNN
accuracy_test = []
accuracy_train = []
Train_Time = []

for i in range(1,99):
    train_size = i/100
    train_x, test_x, train_y, test_y = train_test_split(irisX, irisY,  random_state=0, test_size=1 - i / 100)
    KNN = KNeighborsClassifier(n_neighbors = 8, weights = 'uniform',)
    TT = time.time()
    KNN.fit(x_train, y_train)
    TT = time.time() - TT
    y_pred_test = KNN.predict(x_test)
    y_pred_train = KNN.predict(x_train)
    baccuracy_test = accuracy_score(y_test, y_pred_test)
    baccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(baccuracy_test)
    accuracy_train.append(baccuracy_train)
    Train_Time.append(TT)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.title("KNN Training and Testing - Weighted Uniform - iris")
plt.legend()
plt.savefig("KNN_UniformTT.png")
plt.clf()

plt.plot(Train_Time)
plt.xlabel("Train Size")
plt.ylabel("Training Time")
plt.title("KNN Training Time - neighbor 8 - Cancer dataset ")
plt.legend()
plt.savefig("KNNTT.png")
plt.clf()


#Neural Network

accuracy_test = []
accuracy_train = []

Train_Time = []
for i in range(1,50):
    train_size = i/50
    train_x, test_x, train_y, test_y = train_test_split(irisX, irisY, random_state=0, test_size=1 - i / 50)

    NN = MLPClassifier(hidden_layer_sizes=(25,25,25))
    TT = time.time()
    NN.fit(x_train, y_train)
    TT = time.time() - TT
    y_pred_test = NN.predict(x_test)
    y_pred_train = NN.predict(x_train)
    nnaccuracy_test = accuracy_score(y_test, y_pred_test)
    nnaccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(nnaccuracy_test)
    accuracy_train.append(nnaccuracy_train)
    Train_Time.append(TT)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Train size")
plt.ylabel("Accuracy")
plt.title("Neural Network Training and Testing -- iris")
plt.legend()
plt.savefig("NN - 100 TT.png")
plt.clf()

plt.plot(Train_Time)
plt.xlabel("Train Size (1/50)")
plt.ylabel("Training Time")
plt.title("Neural Network Training Time - iris ")
plt.legend()
plt.savefig("NNTT.png")
plt.clf()


# SVM

accuracy_test = []
accuracy_train = []
Train_Time = []
for i in range(1, 20):
    train_size = i / 20
    train_x, test_x, train_y, test_y = train_test_split(irisX, irisY, random_state=0, test_size=1 - i / 20)
    TT = time.time()
    SVM = svm.SVC(kernel = 'linear')
    SVM.fit(x_train, y_train)
    TT = time.time() - TT
    y_pred_test = SVM.predict(x_test)
    y_pred_train = SVM.predict(x_train)
    saccuracy_test = accuracy_score(y_test, y_pred_test)
    saccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(saccuracy_test)
    accuracy_train.append(saccuracy_train)
    Train_Time.append(TT)


plt.plot(Train_Time)
plt.xlabel("Train Size")
plt.ylabel("Training Time")
plt.title("SVM Train time - iris ")
plt.legend()
plt.savefig("SVMTT.png")
plt.clf()


