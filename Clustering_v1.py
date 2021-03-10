# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:21:41 2021

@author: Meekey
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pandas_profiling
import itertools
import scipy
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer
from kmodes.kmodes import KModes
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# This will ensure that matplotlib figures don't get cut off when saving with savefig()
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


df = pd.read_csv("C:\\Users\\Meekey\\Documents\\GitHub\\Projects\\Mall_Customers.csv")

df = df.rename(index=str, columns={"Genre": "Male"})
df['Male'] = df['Male'].replace(['Male', 'Female'], [True, False])

#convert Annual Income from thousands to hundreds 
df['AnnualIncome'] = 1000* df['AnnualIncome'] 

list(df)
df.shape
df.info()
df.describe().transpose()
df.head(n=20)
df.tail()

X = df.copy()
X.head()
X = X.drop(['CustomerID', 'Age', 'Male'], axis=1) #axis 0 to drop rows
X.head(10)

plt.style.use('default');

plt.figure(figsize=(16, 10));
plt.grid(True);

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c="black", s=200);
plt.title("Mall Data w/o scaling", fontsize=20);
plt.xlabel('Annual Income (K)', fontsize=22);
plt.ylabel('Spending Score', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

#Plot the two metrics that are used for clustering
import matplotlib.pyplot as plt
 
lb=["AnnualIncome","SpendingScore"]                   
box_plot_data=[df["AnnualIncome"],df["SpendingScore"]]
plt.boxplot(box_plot_data,labels=lb, vert=1)
plt.show()

#Plot the two clustering metrics using log scale to cope with the varying scales
import numpy;

box_plot_data=[numpy.log10(df["AnnualIncome"]),numpy.log10(df["SpendingScore"])]
plt.boxplot(box_plot_data,labels=lb, vert=1)
plt.show()

#Scale both clustering metrics to be on the same scale
scaler = StandardScaler()
features = ['AnnualIncome', 'SpendingScore']
X[features] = scaler.fit_transform(X[features])

box_plot_data=[X["AnnualIncome"],X["SpendingScore"]]
plt.boxplot(box_plot_data,labels=lb, vert=1)
plt.show()

#Plot the Data
plt.style.use('default');
plt.figure(figsize=(16, 10));
plt.grid(True);
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c="black", s=200);
plt.title("Mall Data", fontsize=20);
plt.xlabel('Annual Income (K)', fontsize=22);
plt.ylabel('Spending Score', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

#K-Means
k_means = KMeans(init='k-means++', n_clusters=7, n_init=10, random_state=42, n_jobs=-1) #### n_jobs 
k_means.fit(X)

k_means.labels_

# Let's look at the centers
k_means.cluster_centers_
silhouette_score(X, k_means.labels_)

#Plot the Clusters
plt.style.use('default');
plt.figure(figsize=(16, 10));
plt.grid(True);
sc = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=200, c=k_means.labels_);
# plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=500, c="black")
plt.title("K-Means (K=5)", fontsize=20);
plt.xlabel('Annual Income (K)', fontsize=22);
plt.ylabel('Spending Score', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);


for label in k_means.labels_:
    plt.text(x=k_means.cluster_centers_[label, 0], y=k_means.cluster_centers_[label, 1], s=label, fontsize=32, 
             horizontalalignment='center', verticalalignment='center', color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.02));4

def PrintLabels (k_means):
    for label in k_means.labels_:
        plt.text(x=k_means.cluster_centers_[label, 0], y=k_means.cluster_centers_[label, 1], s=label, fontsize=32, 
                 horizontalalignment='center', verticalalignment='center', color='black',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.02));
    return

PrintLabels (k_means)
#plt.savefig('out/mall-kmeans-5.png');

#Internal Validation Metrics
k_means.inertia_
silhouette_score(X, k_means.labels_)


plt.style.use('default');

sample_silhouette_values = silhouette_samples(X, k_means.labels_)
import math;
sizes =200*sample_silhouette_values

plt.figure(figsize=(16, 10));
plt.grid(True);

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=sizes, c=k_means.labels_)
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=500, c="black")

plt.title("K-Means (Dot Size = Silhouette Distance)", fontsize=20);
plt.xlabel('Annual Income (K)', fontsize=22);
plt.ylabel('Spending Score', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

# plt.savefig('out/mall-kmeans-5-silhouette-size.png');

visualizer = SilhouetteVisualizer(k_means)
visualizer.fit(X)
visualizer.poof()
fig = visualizer.ax.get_figure()
# fig.savefig('out/mall-kmeans-5-silhouette.png', transparent=False);


# Instantiate the clustering model and visualizer
visualizer = InterclusterDistance(k_means)
visualizer.fit(X) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data
# plt.savefig('out/mall-kmeans-5-tsne.png', transparent=False);

# Elbow Method (Manual)
inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, max_iter=1000, random_state=42).fit(X)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(X, kmeans.labels_, metric='euclidean')
    

plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");
plt.savefig('out/mall-kmeans-elbow-interia.png');


plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");
# plt.savefig('out/mall-kmeans-elbow-silhouette.png');

# Elbow Method (Using Yellowbrick Package)
model = KMeans(init='k-means++', n_init=10, max_iter=1000, random_state=42)
KElbowVisualizer(model, k=(2,11), metric='silhouette', timings=False).fit(X).poof();
#KElbowVisualizer(model, k=(2,11), metric='calinski_harabaz', timings=False).fit(X).poof();
KElbowVisualizer(model, k=(2,11), metric='distortion', timings=False).fit(X).poof();


# Intepretting the Clusters
# Means
k_means.cluster_centers_

for label in set(k_means.labels_):
    print('\nCluster {}:'.format(label))
    X_tmp = X[k_means.labels_==label].copy()
    X_tmp.loc['mean'] = X_tmp.mean()
    X_tmp.tail(13)


# Find Examplars
from scipy.spatial import distance

for i, label in enumerate(set(k_means.labels_)):    
    X_tmp = X[k_means.labels_==label].copy()
    
    exemplar_idx = distance.cdist([k_means.cluster_centers_[i]], X_tmp).argmin()
    exemplar = pd.DataFrame(X_tmp.iloc[exemplar_idx])
   
    print('\nCluster {}:'.format(label))
    exemplar.transpose()

# Look at Individual Silhouette Scores
k_means = KMeans(init='k-means++', n_clusters=6, n_init=10, random_state=42)
k_means.fit(X)
sample_silhouette_values = silhouette_samples(X, k_means.labels_)
X_tmp = X.copy()
X_tmp['Cluster ID'] = k_means.labels_

silhouette_score(X, k_means.labels_)

X_tmp['Silhouette'] = sample_silhouette_values
X_tmp = X_tmp.sort_values(['Silhouette'])
X_tmp.head()
X_tmp.tail()

plt.figure(figsize=(16, 10));
plt.grid(True);

plt.scatter(X_tmp['AnnualIncome'], X_tmp['SpendingScore'], sizes=200*(X_tmp['Silhouette']+0.05), c=X_tmp['Cluster ID'])

plt.title("K-Means (K = {}, Sil={:.2f})".format(k_means.n_clusters, silhouette_score(X, k_means.labels_, metric='euclidean')), fontsize=20);
plt.xlabel('Annual Income (K)', fontsize=22);
plt.ylabel('Spending Score', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

for label in k_means.labels_:
    plt.text(x=k_means.cluster_centers_[label, 0], y=k_means.cluster_centers_[label, 1], s=label, fontsize=32, 
             horizontalalignment='center', verticalalignment='center', color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.02));
    
for i, txt in enumerate(X_tmp.index.values):
    sil = X_tmp['Silhouette'].iloc[i]
    if sil < 0.05:
        plt.text(x=X_tmp['AnnualIncome'].iloc[i], y=X_tmp['SpendingScore'].iloc[i], s=txt, fontsize=22)

# plt.savefig('out/mall-kmeans-{}-silhouette-size-labels.png'.format(k_means.n_clusters));


visualizer = SilhouetteVisualizer(k_means)
visualizer.fit(X)
visualizer.poof()
fig = visualizer.ax.get_figure()
# fig.savefig('out/mall-kmeans-{}-silhouette-f.png'.format(k_means.n_clusters), transparent=False);

# DBSCAN
db = DBSCAN(eps=0.2, min_samples=2)
db.fit(X)
db.labels_
silhouette_score(X, db.labels_)
plt.figure(figsize=(16, 10));
plt.grid(True);

unique_labels = set(db.labels_)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))];

for k in unique_labels:
    if k == -1:        # Black used for noise.
        col = [0, 0, 0, 1]
    else:
        col = colors[k]

    xy = X[db.labels_ == k]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14);

    
plt.title('');
plt.title("DBSCAN (n_clusters = {:d}, black = outliers)".format(len(unique_labels)), fontsize=20);
plt.xlabel('Annual Income (K)', fontsize=22);
plt.ylabel('Spending Score', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);
# plt.savefig('out/mall-dbscan-03.png', transparent=False);

silhouettes = {}
for eps in np.arange(0.1, 0.6, 0.1):
    db = DBSCAN(eps=eps, min_samples=3).fit(X)
    silhouettes[eps] = silhouette_score(X, db.labels_, metric='euclidean')
    

plt.figure();
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('DBSCAN, Elbow Method')
plt.xlabel("Eps");
plt.ylabel("Silhouette");
plt.grid(True);
# plt.savefig('out/mall-dbscan-03-silhouette.png', transparent=False);


# Parameter Exploration
def do_dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    unique_labels = set(db.labels_)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        print('eps={}, min_samples={}, n_clusters <= 1. Returning.'.format(eps, min_samples))
        return
    
    sil = silhouette_score(X, db.labels_)
    print("eps={}, min_samples={}, n_clusters={}, sil={}".format(eps, min_samples, n_clusters, sil))
    
    plt.figure(figsize=(16, 10));
    plt.grid(True);   
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))];

    for k in unique_labels:
        if k == -1:        # Black used for noise.
            col = [0, 0, 0, 1]
        else:
            col = colors[k]

        xy = X[db.labels_ == k]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14);


    plt.title('');
    plt.title("DBSCAN (eps={}, min_samples={}, n_clusters = {:d}, sil={:.3f})".format(eps, min_samples, n_clusters, sil), fontsize=20);
    plt.xlabel('Annual Income (K)', fontsize=22);
    plt.ylabel('Spending Score', fontsize=22);
    plt.xticks(fontsize=18);
    plt.yticks(fontsize=18);
    plt.savefig('out/mall-dbscan-auto-{}-{}.png'.format(eps, min_samples), transparent=False);
    

epss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
min_samples = range(1, 10)

for prod in list(itertools.product(epss, min_samples)):
    do_dbscan(X, prod[0], prod[1])



























