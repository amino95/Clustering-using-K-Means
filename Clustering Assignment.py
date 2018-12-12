
# coding: utf-8

# # Clustering Using Scikit-Learn Library
# ## Part 1 : kMeans Clustering
# 
# 

# ### Description of the dataset
# The dataset we will use contains preprocessed/clean data about **440 clients** of a wholesale distributor, mainly about the products that clients buy during one year. Below is a brief description of the **8 attributes** in this dataset:
# 
# - FRESH: annual spending on fresh products
# - MILK: annual spending on milk products
# - GROCERY: annual spending on grocery products
# - FROZEN: annual spending on frozen products
# - DETERGENTS_PAPER: annual spending on detergents and paper products
# - DELICATESSEN: annual spending on delicatessen products
# - TYPE: type of customer - Hotel/Restaurant/Cafe (1) or wholesale depot (2)
# - REGION: region where customer lives - Tunis (1), Sfax (2) or Other (3)
# 
# We will use this dataset to extract clustering patterns, i.e. determine whether these customers can be divided into a small number of groups.
# 
# Source: http://archive.ics.uci.edu/ml/datasets/Wholesale+customers

# In[1]:


# import Python libraries we will need later

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


# allow plots to appear inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# set the font size of plots
plt.rcParams['font.size'] = 14


# In[3]:


# import some modules from the scikit-learn library

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ***

# ## Get the data

# In[4]:


df0 = pd.read_csv('Wholesale customers data.csv')


# In[5]:


print(type(df0))
print(df0.shape)


# In[6]:


df0.head()


# In[7]:


df0.dtypes


# In[8]:


df0.describe()


# In[9]:


df0.Type.value_counts()


# In[10]:


df0.Region.value_counts()


# *******

# ## Data Preparation

# In[38]:


get_ipython().run_line_magic('pinfo', 'MinMaxScaler')


# In[32]:


# Create a copy of original data frame
df_hot = df0


# ###  a) One hot encoding

# In[33]:


#  Convert the 'Type' attribute to binary using the get_dummies() method
dummies = pd.get_dummies(df_hot['Type'], prefix='Type')  #create 2 binary attributes based on Type attrubute
df_hot = pd.concat([df_hot, dummies], axis=1)    #add these attrubutes to data frame
df_hot.drop('Type', axis=1, inplace=True)  # remove the categorical attribute from data frame
df_hot.head()


# In[34]:


# Remove the Type_2 attribute because it is redundant
df_hot.drop('Type_2', axis=1, inplace=True)
df_hot.head()


# In[35]:


# Do the same thing for the other categorical attributes
# WRITE YOUR CODE BELOW
dummies = pd.get_dummies(df_hot['Region'], prefix='Region')  #create 2 binary attributes based on Type attrubute
df_hot = pd.concat([df_hot, dummies], axis=1)    #add these attrubutes to data frame
df_hot.drop('Region', axis=1, inplace=True)  # remove the categorical attribute from data frame
df_hot.head()


# In[36]:


df_hot.drop('Region_3', axis=1, inplace=True)
df_hot.head()


# In[37]:


# verify the shape of the new data frame
df_hot.shape


# ### b) Scale normalization
# In order to give equal importance to all our attributes, we are going to transform all attributes to the range [0,1].
# We will do this using the``MinMaxScaler`` class.
# 
# **Reference**: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

# In[61]:


# WRITE YOUR CODE BELOW (try to follow the comments)

# create an instance of MinMaxScaler class
scaler = MinMaxScaler()

# call the fit() method
scaler.fit(df_hot)

# call the transform() method
scaler = scaler.transform(df_hot)

# put the result in a data frame called df_transformed
df_transformed = pd.DataFrame(scaler)

print(df_transformed)


# In[62]:


# Check the size of the new data frame
df_transformed.shape


# In[63]:


# Check the first few lines in the new data frame
df_transformed.head()


# In[64]:


# Check the distribution of values (make sure they are between 0 and 1)
df_transformed.describe()


# ******

# ## kMeans Clustering Method

# **References: **
# - http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# - http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

# In[79]:


# Read documentation of KMeans class constructor
get_ipython().run_line_magic('pinfo', 'KMeans')


# In[72]:


# Run kMeans with 3 clusters
K = 3
km = KMeans(n_clusters=K, random_state=10)   # create instance of KMeans class
km.fit(df_transformed)  # run kMeans algorithm with 3 clusters


# In[73]:


# Determine the value of SSD (Sum of Squared Distances)
km.inertia_


# In[82]:


# How many objects are there in each cluster?
# Hint: use km.labels_ (member variable of KMeans class)
# WRITE YOUR CODE BELOW
pd.Series(km.labels_).value_counts()


# In[83]:


# Run kMeans algorithm with different values of K: K=1,...,15  and store the value of SSD for each K in a list
# COMPLETE THE CODE BELOW

K_values = range(1,16)
Sum_of_squared_distances = []   # initialize empty list to store SSD values
for k in K_values:
    km = KMeans(n_clusters=k, random_state=10)   # create instance of KMeans class
    km.fit(df_transformed)
    x = km.inertia_
    Sum_of_squared_distances.append(x)


# In[84]:


# Plot Sum_of_squared_distances vs. K_values
# Hint: use the function plt.plot() ...
# WRITE YOUR CODE BELOW
plt.plot(Sum_of_squared_distances)


# In[85]:


# Re-run kMeans with the best value of K
# WRITE YOUR CODE BELOW
K = 5
km = KMeans(n_clusters=K, random_state=10)   # create instance of KMeans class
km.fit(df_transformed)


# In[86]:


km.inertia_


# In[90]:


# Calculate the silhouette coefficient by calling the function silhouette_score() imported at the beginning of this file
# Reference: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# WRITE YOUR CODE BELOW
scores = silhouette_score(df_transformed, km.labels_)
print(scores)


# ****
