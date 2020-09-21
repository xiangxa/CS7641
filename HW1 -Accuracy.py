
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


irisX,irisY = load_iris(return_X_y=True)
cancerx,cancery = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(cancerx,cancery, test_size=0.2, random_state=1)
#Decision Tree
defaultDT = tree.DecisionTreeClassifier()


DDT = defaultDT.fit(x_train, y_train)

DT = tree.DecisionTreeClassifier(min_samples_split=7)
DT10= tree.DecisionTreeClassifier(min_samples_split=100)
DT = DT.fit(x_train, y_train)
DT10 = DT10.fit(x_train, y_train)

y_pred = DT.predict(x_test)
y_10pred = DT10.predict(x_test)
dy_pred = DDT.predict(x_test)
dtaccuracy = accuracy_score(y_test,y_pred)
dt10accuracy = accuracy_score(y_test,y_10pred)
ddtaccuracy = accuracy_score(y_test,dy_pred)
print(dtaccuracy, dt10accuracy, ddtaccuracy )

accuracy_test = []
accuracy_train = []
for i in range(2,100):
    DTP = tree.DecisionTreeClassifier(min_samples_split=i)
    DTP = DTP.fit(x_train, y_train)
    y_predp_test = DTP.predict(x_test)
    y_predp_train = DTP.predict(x_train)
    dtpaccuracy_test = accuracy_score(y_test, y_predp_test)
    dtpaccuracy_train = accuracy_score(y_train, y_predp_train)
    accuracy_test.append(dtpaccuracy_test)
    accuracy_train.append(dtpaccuracy_train)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Min sample Split")
plt.ylabel("Accuracy")
plt.title("Decision Tree of Training and Testing - IRIS ")
plt.legend()
plt.savefig("DTAccuracy.png")
plt.clf()


######



#Boost 
accuracy_test = []
accuracy_train = []
baccuracy_test = []
baccuracy_train = []
for i in range(2,50):
    DTP = tree.DecisionTreeClassifier(min_samples_split=i)
    DTP.fit(x_train, y_train)
    Boost = AdaBoostClassifier(base_estimator=DTP,n_estimators=50,random_state=0)
    Boost.fit(x_train, y_train)
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

plt.plot(accuracy_test, label='DTtest')
plt.plot(accuracy_train, label='DTtrain')
plt.plot(baccuracy_test, label='Boosttest')
plt.plot(baccuracy_train, label='Boosttrain')

plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.title("Boosting Decision Tree Training and Testing - 50 estimators - IRIS")
plt.legend()
plt.savefig("DDTBoostAccuracy.png")
plt.clf()



#KNN
accuracy_test = []
accuracy_train = []
for i in range(1,50):
    KNN = KNeighborsClassifier(n_neighbors = i, weights = 'uniform',)
    KNN.fit(x_train, y_train)
    y_pred_test = KNN.predict(x_test)
    y_pred_train = KNN.predict(x_train)
    baccuracy_test = accuracy_score(y_test, y_pred_test)
    baccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(baccuracy_test)
    accuracy_train.append(baccuracy_train)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("KNN Training and Testing - Weighted Uniform - IRIS")
plt.legend()
plt.savefig("KNN_Uniform.png")
plt.clf()


#Neural Network

accuracy_test = []
accuracy_train = []
tup = (100,)
for i in range(1,50):
    tup = tup + (100,)
    NN = MLPClassifier(hidden_layer_sizes=tup)
    NN.fit(x_train, y_train)
    y_pred_test = NN.predict(x_test)
    y_pred_train = NN.predict(x_train)
    nnaccuracy_test = accuracy_score(y_test, y_pred_test)
    nnaccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(nnaccuracy_test)
    accuracy_train.append(nnaccuracy_train)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Number of Layers")
plt.ylabel("Accuracy")
plt.title("Neural Network Training and Testing -- 100 units  - IRIS")
plt.legend()
plt.savefig("NN - 100.png")
plt.clf()

#Neural Network

accuracy_test = []
accuracy_train = []
tup = (50,)
for i in range(1,50):
    tup = tup + (50,)
    NN = MLPClassifier(hidden_layer_sizes=tup)
    NN.fit(x_train, y_train)
    y_pred_test = NN.predict(x_test)
    y_pred_train = NN.predict(x_train)
    nnaccuracy_test = accuracy_score(y_test, y_pred_test)
    nnaccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(nnaccuracy_test)
    accuracy_train.append(nnaccuracy_train)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Number of Layers")
plt.ylabel("Accuracy")
plt.title("Neural Network Training and Testing -- 50 units - IRIS")
plt.legend()
plt.savefig("NN - 50.png")
plt.clf()

#Neural Network

accuracy_test = []
accuracy_train = []
tup = (25,)
for i in range(1,50):
    tup = tup + (25,)
    NN = MLPClassifier(hidden_layer_sizes=tup)
    NN.fit(x_train, y_train)
    y_pred_test = NN.predict(x_test)
    y_pred_train = NN.predict(x_train)
    nnaccuracy_test = accuracy_score(y_test, y_pred_test)
    nnaccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(nnaccuracy_test)
    accuracy_train.append(nnaccuracy_train)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Number of Layers")
plt.ylabel("Accuracy")
plt.title("Neural Network Training and Testing -- 25 units - IRIS")
plt.legend()
plt.savefig("NN - 25.png")
plt.clf()



accuracy_test = []
accuracy_train = []

for i in range(1,50):
    NN = MLPClassifier(hidden_layer_sizes=(i,))
    NN.fit(x_train, y_train)
    y_pred_test = NN.predict(x_test)
    y_pred_train = NN.predict(x_train)
    nnaccuracy_test = accuracy_score(y_test, y_pred_test)
    nnaccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(nnaccuracy_test)
    accuracy_train.append(nnaccuracy_train)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Number of Units")
plt.ylabel("Accuracy")
plt.title("Neural Network Training and Testing -- 1 layer - IRIS")
plt.legend()
plt.savefig("NN - 1 layer.png")
plt.clf()


accuracy_test = []
accuracy_train = []

for i in range(1,50):
    NN = MLPClassifier(hidden_layer_sizes=(50,50,i))
    NN.fit(x_train, y_train)
    y_pred_test = NN.predict(x_test)
    y_pred_train = NN.predict(x_train)
    nnaccuracy_test = accuracy_score(y_test, y_pred_test)
    nnaccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(nnaccuracy_test)
    accuracy_train.append(nnaccuracy_train)

plt.plot(accuracy_test, label='test')
plt.plot(accuracy_train, label='train')
plt.xlabel("Number of Units")
plt.ylabel("Accuracy")
plt.title("Neural Network Training and Testing -- 3 layers - IRIS")
plt.legend()
plt.savefig("NN - 3 layer.png")
plt.clf()

# SVM

accuracy_test = []
accuracy_train = []
for i in ['linear', 'rbf', 'sigmoid']:
    SVM = svm.SVC(kernel =i)
    SVM.fit(x_train, y_train)
    y_pred_test = SVM.predict(x_test)
    y_pred_train = SVM.predict(x_train)
    saccuracy_test = accuracy_score(y_test, y_pred_test)
    saccuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test.append(saccuracy_test)
    accuracy_train.append(saccuracy_train)
    print (saccuracy_test, saccuracy_train)

#plt.plot(accuracy_test, label='test')
#plt.plot(accuracy_train, label='train')
#plt.xlabel("kernel")
#plt.ylabel("Accuracy")
#plt.title("SVM Training and Testing  - Cancer Dataset")
#plt.legend()
#plt.savefig("SVM Accuracy.png")
#'plt.clf()

