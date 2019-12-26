# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:30:05 2019

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
plt.rcParams['figure.figsize']=(12,7)
dataset = pd.read_csv('driver-data.csv')

print (dataset.head())
dataset.info()
print (dataset.describe())
x = dataset.iloc[:,[1,2]].values

from sklearn.cluster import KMeans

# elbow method
wcss = []
for i in range(1,11):                                 #number of clusters
    kmeans = KMeans(n_clusters=i, init = "k-means++") #init-generates centroids in the dataset based on the density
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print (wcss)                                          #list of distances

plt.plot(range(1,11),wcss, c='purple')
plt.xlabel("cluster")
plt.ylabel("wcss")
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++')
y_kmeans = kmeans.fit_predict(x)                        #cluster no. of individual drivers

# cluster center vectors
kmeans.cluster_centers_

print (y_kmeans)
print (len(y_kmeans))

# check how many drivers are there in 1st and 2nd cluster

print (type(y_kmeans))
unique, counts = np.unique(y_kmeans,return_counts=True)
print (dict(zip(unique,counts)))

plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1],s=100,c='blue',label='Cluster1')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1],s=100,c='red',label='Cluster2')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1],s=100,c='yellow',label='Cluster3')
plt.scatter(x[y_kmeans==3,0], x[y_kmeans==3,1],s=100,c='violet',label='Cluster4')


plt.legend()
plt.xlabel('mean_dist_day')
plt.ylabel('mean_over_speed_perc')
plt.show()

clust = pd.DataFrame(y_kmeans)
clust.columns = ['Cluster']

dataset=pd.concat([dataset,clust],axis=1)    #concat table by index
dataset['Cluster'].replace(0,'Group 1',inplace=True)
dataset['Cluster'].replace(1,'Group 2',inplace=True)
dataset['Cluster'].replace(2,'Group 3',inplace=True)
dataset['Cluster'].replace(3,'Group 4',inplace=True)
'''dataset.to_csv(r'C:\Users\user\Desktop\Driver Data With 4 Cluster Groups.csv')'''
# create hierarchy model

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))   #creating dendrogram
plt.title('Dendrogram')
plt.xlabel('Drivers')
plt.ylabel('Euclidean Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')     #creating cluster
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc==0,0], x[y_hc==0,1], c='red', s=100, label='Cluster1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], c='blue', s=100, label='Cluster2')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1], c='yellow', s=100, label='Cluster3')
plt.scatter(x[y_hc==3,0], x[y_hc==3,1], c='violet', s=100, label='Cluster4')

plt.xlabel('mean_dist_day')
plt.ylabel('mean_over_speed_perc')
plt.legend()
plt.show()





