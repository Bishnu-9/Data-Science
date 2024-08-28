#!/usr/bin/env python
# coding: utf-8

# # A. Problem Statement
# 
# An online retail store is trying to understand the various customer purchase patterns for their firm, you are required to give enough evidence based insights to provide the same.

# # B. Project Objective
# 
# The objective of this project is to group the customers according to their purchase behaviour so that store can plan out different strategies for different customers.

# # C. Data Discription
# 
# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('OnlineRetail (3).csv',encoding='unicode_escape')


# In[3]:


df


# # D. Data Pre-processing Steps and Inspiration
# The Pre-processing of the data includes the following steps:
# 
# Data Cleaning: Cleaning the data by removing missing values and other inconsistencies.
# Data Exploration: Exploring the data to gain insights and understanding the data.
# Data Visualization: Visualizing the data for better understanding.

# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[ ]:


# Since our analysis is focused on customer segmentation, records without CustomerID cannot be used for clustering


# In[7]:


df = df.dropna()
df.isnull().sum()


# In[11]:


plt.figure(figsize=(12, 6))
customer_count = df.groupby('InvoiceNo')['Description'].count().nlargest(20)
customer_count.plot(kind='bar', color='blue')
plt.xlabel('InvoiceNo')
plt.ylabel('No of products')
plt.title('Number of products per InvoiceNo')
plt.xticks(rotation=90)
plt.show()


# In[10]:


top_CustomerID_quantity = df.groupby('CustomerID')['Quantity'].sum().nlargest(20)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_CustomerID_quantity.index, y=top_CustomerID_quantity.values)
plt.xticks(rotation=90)
plt.title("Quantity per CustomerID")
plt.xlabel("CustomerID")
plt.ylabel("Quantity")
plt.show()


# In[26]:


df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


# In[13]:


customer_df = df.groupby('CustomerID').agg({'InvoiceNo': 'nunique', 'TotalPrice': 'sum',  'Quantity': 'sum'  }).reset_index()


# In[14]:


customer_df


# In[15]:


plt.figure(figsize=(8, 6))
sns.heatmap(customer_df.corr(), annot=True, cmap='coolwarm')
plt.show()


# # Observations from the heatmap:
# 
# 1. CustomerId has no correlation with other features.
# 
# 2. Invoice and TotalPrice have a moderate positive correlation of 0.57, suggesting that customers with more transactions tend to have a higher total spending.
# 
# 3. TotalPrice and Quantity have a very strong positive correlation of 0.92, indicating that as the quantity of products purchased increases, the total price tends to increase as well.
# 
# 4. TransactionCount and Quantity also have a moderate positive correlation of 0.57, implying that customers who buy more items also tend to make more transactions

# In[17]:


scaler = StandardScaler() # standardizing the data 


# In[18]:


numerical_features = ['InvoiceNo', 'TotalPrice', 'Quantity']


# In[19]:


customer_df[numerical_features] = scaler.fit_transform(customer_df[numerical_features])


# In[20]:


customer_df.head()


# # E. Choosing the Algorithm for the Project
# 
# The choice of algorithm for a machine learning project is depends upon the type of problem we are trying to solve. Generally, supervised learning algorithms are used for classification and regression problems, while unsupervised learning algorithms are used for clustering and dimensionality reduction tasks. Some popular algorithm used for customer segmentation are K means clustering , Heirarchical Clustering, DBscan etc.

# In[21]:


from sklearn.cluster import KMeans


# In[22]:


inertia = []
range_values = range(1, 10)


# In[27]:


for i in range_values:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(customer_df[numerical_features])
    inertia.append(kmeans.inertia_)


# In[24]:


plt.plot(range_values, inertia, 'o-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# In[35]:


kmeans = KMeans(n_clusters= 3, random_state=42)
kmeans.fit(customer_df[numerical_features])


# In[36]:


customer_df['Cluster'] = kmeans.labels_


# In[37]:


print(customer_df['Cluster'].value_counts())


# In[38]:


cluster_analysis = customer_df.groupby('Cluster')[numerical_features].mean()
print(cluster_analysis)


# # F. Motivation and Reasons For Choosing the Algorithm¶
# 
# Here we could have used K means or heirarchical clustering.But as the data set is large we prefer to go with k Means clustering. Since we are using more than 2 features we are using PCA to reduce to 2D for visualization purposes

# In[39]:


from sklearn.decomposition import PCA


# In[40]:


pca = PCA(n_components=2)
customer_df['pc1'], customer_df['pc2'] = zip(*pca.fit_transform(customer_df[numerical_features]))


# In[48]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='pc1', y='pc2', hue='Cluster', data=customer_df, palette='Set1')
plt.title('Cluster Visualization')
plt.show()


# # G. Inferences from the Same
# 
# 1. Customers in Cluster 0 are the core customer group with average spending habits and might be targeted with general marketing strategies.
# 
# 2. Customers in Cluster 1 are valuable with potentially higher spending and may be responsive to promotional activities.
# 
# 3. Customers in Cluster 2 are likely the most valuable and could possibly be the focus of premium services, exclusive offers.
# 

# # H.Future Possibilities of the Project
# 
# This customer segmentation in all possibilities is going to help the retail shop to understand their customer and strategise their marketing accordingly. In future with more data avalaibility we could use the RMFT model for enhancement.
