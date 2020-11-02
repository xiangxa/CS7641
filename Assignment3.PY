import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from sklearn.random_projection import GaussianRandomProjection

from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer


x_dia_origin, y_diabetes = load_diabetes(return_X_y=True)


x_cancer_origin, y_cancer= load_breast_cancer(return_X_y=True)


#normalized

x_diabetes = MinMaxScaler().fit_transform(x_dia_origin)
x_cancer = MinMaxScaler().fit_transform(x_cancer_origin)



## elbow figure
def elbow(k,x,dataset,filename):
    k_range = list(range(1, k))

    for k in k_range:
        km = KMeans(n_clusters=k)
        km.fit(x)
        error.append(km.inertia_)

    plt.figure(figsize=(8, 8))
    plt.plot(k_range, error, "o-", linewidth=3)
    plt.xlim(xmin=0, xmax=k)
    plt.title(f'Sum Square Error: {dataset}',)
    plt.xlabel("the number of cluster",  fontsize=16)
    plt.ylabel('sum of squared error', fontsize=16)
    plt.savefig(filename)
    
elbow(14,x_diabetes,"Diabetes","ssr_Diabetes.png")
elbow(14,x_cancer,"Breast Cancer Dataset","ssr_cancer.png")

def EM(x1,x2,x3):
    components = np.arange(1, 25)
    model = [mixture.GaussianMixture(x, covariance_type='full', random_state=999).fit(x1) for x in components]

    plt.figure(figsize=(8, 8))
    plt.plot(components, [y.bic(x1) for y in model], label='BIC', linewidth=6)

    plt.title(f'EM algorithm cluster: {x2}', fontsize=20)
    plt.legend(loc='best')
    plt.xlabel('K Value', fontsize=16)
    plt.savefig(x3)

EM(x_diabetes,"diabetes","EMdiabetes.png")

EM(x_cancer,"Breast Cacancer","EM.Cancer.png")



def pca1(x1,x2,x3,y):
    pca = PCA().fit(x1)

    j=[]
    for i in temp2:
        model=PCA(i)
        s=model.fit_transform(x1)
        j.append(error(s,y))
    plt.figure(figsize=(8, 8))
    plt.plot(temp2,temp1,linewidth=6,label="variance ratio")

    plt.xlabel('the number of principal component', fontsize=16)

    plt.title(f'PCA: {x2}', fontsize=20)
    plt.legend(loc='best')

    plt.xlim(xmax=temp1.size+1)
    plt.savefig(x3)

pca1(x_diabetes,"Diabetes","PCADiabetes.png",y_diabetes)
pca1(x_cancer,"Cancer","PCACancer.png",y_cancer)

pca = PCA(0.90)
pca.data = pca.fit_transform(x_cancer)
elbow(15,pca.data,"cancer_fit_pca.png","cancer_fit_pca.png")

EM(pca.data,"Cancer fit PCA EM","EMCancerPCA.png")

pca = PCA(0.99)
pca.data = pca.fit_transform(x_diabetes)
elbow(15,pca.data,"Diabetes fit PCA","diafit PCA.png")

EM(pca.data,"Diabetes fit PCA","EM_Diabetes_PCA.png")



def ica(data,data2,output,y):

    j=[]
    for i in range(1,13):
        ica = FastICA(n_components=i)
        ica.data = ica.fit_transform(data)
        j.append(error(ica.data,y))
    plt.figure(figsize=(8, 8))

    plt.xlabel('the number of independent component', fontsize=16)
    plt.ylabel('error rate', fontsize=16)
    plt.title(f'ICA: {data2}', fontsize=20)
    plt.savefig(output)




ica(x_cancer,"Breast Cancer","ICAcancer.png",y_cancer)
ica(x_diabetes,"Diabetes","ICA.diab.png", y_diabetes)

ica = FastICA(10)
ica.data = ica.fit_transform(x_diabetes)

elbow(20,ica.data,"diaelbow.ica.png","Diabete Fit ICA")

EM(ica.data,"Diabites fit ICA","ICADIabetesicaEM.png")
ica = FastICA(4)
ica.data = ica.fit_transform(x_cancer)

elbow(10,ica.data,"cancerICAfit.png","Cancer Fit ICA")

EM(ica.data,"cancerfit ICA","ica4cancerem.png")


def rca(data,data2,output,y):
    plt.figure(figsize=(8, 8))
    for round in range(1,20):
        j = []
        for i in range(1, 10):
            rca = GaussianRandomProjection(n_components=i)
            rca.data = rca.fit_transform(data)
            j.append(error(rca.data, y))
        plt.plot(range(1, 10), j, linewidth=3)
    plt.xlabel('the number of random component', fontsize=16)
    plt.ylabel('error rate', fontsize=16)
    plt.title(f'RP, 20 times: {data2}', fontsize=20)

    plt.savefig(output)


rca(x_cancer,"Breast Cancer","RCAcancer.png",y_cancer)
rca(x_diabetes,"Diabetes","RCA.diab.png",y_diabetes)
