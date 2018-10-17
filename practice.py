# PCA and K-Mean Implementation on wine dataset
# @author : yohanesam

import matplotlib.pyplot as plt             # import plotting library from matplotlib python
from mpl_toolkits.mplot3d import axes3d     # import 3d plotting library from matplotlib python
import numpy as np                          # utility library for python
import pandas as pd                         # library for importing dataset and such
import tensorflow as tf                     # our main computing libraries

# loading our dataset based our path and column name
filename = "dataset/winequality-white.csv"
names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
dataset = pd.read_csv(filename, names=names, sep=';')

# prepare list container to contain necessary data for our 3d plotting
xs = []
ys = []
zs = []
color = []

# create our figure and subploting, so we can combine 2 plot to see
# the diference on our plotting
fig = plt.figure(dpi=100)
plot1 = fig.add_subplot(2, 1, 1, projection='3d')
for i in range(0, len(dataset)) :
    xs.append(dataset.values[i][0])
    ys.append(dataset.values[i][1])
    zs.append(dataset.values[i][2])

    if(dataset.values[i][11] == 3) :
        color.append("tan")

    if(dataset.values[i][11] == 4) :
        color.append("peachpuff")
    
    if(dataset.values[i][11] == 5) :
        color.append("yellowgreen")
    
    if(dataset.values[i][11] == 6) :
        color.append("forestgreen")
        
    if(dataset.values[i][11] <= 7) :
        color.append("navy")


# Plot our data and give the label.
# This is the first part of ploting.
for x, y, z, c in zip(xs, ys, zs, color) :
    plot1.scatter(x, y, z, alpha=0.8, color=c, depthshade=0)

plot1.set_xlabel('fixed acidity')
plot1.set_ylabel('volatile acidity')
plot1.set_zlabel('citric acid')


# Now, we will begin our "PCA dimension reduction" to see the difference
# of our data. We will reduce our 3d to 2d using "Singular Value Decomposition".
# We will doing it mostly using Tensorflow.

# Create a graph, because in Tensorflow every operation is represented as a dataflow graph
data_to_tensor = dataset.iloc[:, 0:11]
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with graph.as_default():
    # Creating a placeholder, because we will feed our dataset to this place holder
    # based on good practice
    ds_matrix = tf.placeholder(tf.float64, shape=data_to_tensor.shape)

    # The SVD in Tensorflow.
    # It will return Σ, U, and V respectively
    s, u, v = tf.svd(ds_matrix)
    
    # The "Diagonal Value" of Σ 
    Sk = tf.diag(s)

# Feed our placeholder with our dataset and then assign it
# to Tensorflow Session
feed_dict = {ds_matrix: data_to_tensor.values}

U, singular_values, sigma = sess.run([u, s, Sk], feed_dict=feed_dict)
    

# Normalize the singular value, and then culative sum the nomalized value
normalized_singular_values = singular_values / sum(singular_values)
ladder = np.cumsum(normalized_singular_values)

# After then, we slice the unnecessary data and leaving the first 2 dimension
with graph.as_default():
    sigma = tf.slice(sigma, [0, 0], [ds_matrix.shape[1], 2])

# Complete the PCA by multipy the U matrice and nomalized Σ matrice
pca = tf.matmul(U, sigma)

# And then we feed throught our data to "PCA algoritm"
pca_to_array = pca.eval(feed_dict=feed_dict)

# Represent the data with plot
data = {"x": [], "y": []}

for x in range(0, len(pca_to_array)) :
    data["x"].append(pca_to_array[x, 0])
    data["y"].append(pca_to_array[x, 1])

df = pd.DataFrame(data)

"""
Ploting after PCA is below, Uncomment this part to see the difference
"""
# plot2 = fig.add_subplot(2, 1, 2)  

# for x, y in zip(data["x"], data["y"]) :
#     plot2.scatter(x, y, alpha=0.8)

# plot2.set_xlabel('fixed acidity')
# plot2.set_ylabel('volatile acidity')


# Now, we create one of the most popular clustering, The great "K-Mean Clustering"
# I dont know if we can implement those algoritm in tensorflow because i dont have so much time :(.
# But dont worry. Fortunately, tensorflow already has class to implement it. i hope it works XD

# convert our dataframe-type dataset into tensor
def input_fn():
    return tf.train.limit_epochs(
        tf.convert_to_tensor(df.values, dtype=tf.float32), num_epochs=1
    )

# set the number of cluster do you want. i suggest 3,
# because the cluster will become mess up when you set more than 3.
# i still dont know why thought :/ tell me if you know :)

# After then, just set the built-in K-Mean cluster like below
num_of_clusters = 3
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_of_clusters,
    use_mini_batch=False
)

# set the number of training session do you wish
num_iterations = 10

# for loop to train based on num_iterations
for _ in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()

# Create a list of cluster node and the color container
# So we can see the difference when plotting our data
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
color = []

for i, point in enumerate(df.values):
    if(cluster_indices[i] == 0) :
        color.append("tan")

    if(cluster_indices[i] == 1) :
        color.append("peachpuff")
    
    if(cluster_indices[i] == 2) :
        color.append("yellowgreen")

# Save our cluster node in cluster dictionary 
cluster = {"x": [], "y": []}

for k in range(len(cluster_centers)) :
    cluster["x"].append(cluster_centers[k, 0])
    cluster["y"].append(cluster_centers[k, 1])
    
plot2 = fig.add_subplot(2, 1, 2)  

"""
Ploting after K-Mean is below
"""

for x, y, c in zip(data["x"], data["y"], color) :
    plot2.scatter(x, y, alpha=0.8, color=c, label=color)

for x, y in zip(cluster["x"], cluster["y"]) :
    plot2.scatter(x, y, marker="x", color="r", label="Cluster poin")

plot2.set_xlabel('fixed acidity')
plot2.set_ylabel('volatile acidity')


# Know what, it work flawlessly.
# But still, this type of code is not perfect, and somewhat bad practice
# because it's taking so much time to run this code.
# I hope i can fix this next time
plt.show()