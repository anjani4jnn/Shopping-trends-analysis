#!/usr/bin/env python
# coding: utf-8

# In[184]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import polars as pl
import pyarrow
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[185]:


df = pd.read_csv('shopping_trends_updated.csv', sep=',')
df.rename(columns={'Purchase Amount (USD)':'Purchase Amount USD'},inplace=True)
print(df.describe())
print(df.describe(include='object'))
print(df.info())
print(df['Review Rating'].unique())
c = np.isinf(df['Age']).values.sum() 
print(str(c)) 


# In[186]:


# distributions of continuous data :
pd.option_context('mode.use_inf_as_na', True)
sns.displot(df['Age'], kde=True)
sns.displot(df['Purchase Amount USD'], kde=True)
sns.displot(df['Review Rating'], kde=True)
sns.displot(df['Previous Purchases'], kde=True)


# In[187]:


# numbers of categorical data :
for col in df.select_dtypes(['object']).columns:
    fig = plt.figure(figsize=(30,10))
    ax = sns.countplot(x=df[col], palette='vlag', hue=df['Gender'], order = df[col].value_counts().index)
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()



# In[188]:


sns.countplot(data=df, y="Location", palette="icefire", order=df["Location"].value_counts().iloc[:10].index)
plt.show()


# In[189]:


sns.countplot(data=df, y="Item Purchased", palette="icefire", order=df["Item Purchased"].value_counts().iloc[:15].index)
plt.show()


# In[190]:


sns.countplot(data=df, y="Color", palette="icefire", order=df["Color"].value_counts().iloc[:15].index)


# In[191]:


print(df[(df["Subscription Status"]=="No") & (df["Discount Applied"]=="Yes")]["Discount Applied"].count())
print(df[(df["Subscription Status"]=="Yes") & (df["Discount Applied"]=="Yes")]["Discount Applied"].count())


# In[192]:


# encode data to make correlations:
df.drop(columns='Customer ID', inplace=True)
encode_df = pd.get_dummies(df, columns=['Gender'], dtype=int)
encode_df['Subscription Status'] = encode_df['Subscription Status'].map({'Yes': 1, 'No': 0})
encode_df['Discount Applied'] = encode_df['Discount Applied'].map({'Yes': 1, 'No': 0})
encode_df['Promo Code Used'] = encode_df['Promo Code Used'].map({'Yes': 1, 'No': 0})


# In[193]:


# spearman's correlation but discrete and binary data:
plt.subplots(figsize=(12,12))
corr_s = encode_df.corr('spearman', numeric_only=True)
sns.heatmap(corr_s, cmap="rainbow", vmin=-0.9, vmax=0.9, fmt='.3f', annot=True)
plt.show()
# columns: 'gender','subscription status', 'discount applied', 'promo code used' seem to be correlated


# In[194]:


# looking for significant correlations with pvalue :
# spearman's and pointbiserial correlation :
print(encode_df[['Purchase Amount USD', 'Previous Purchases', 'Age', 'Review Rating']].corrwith(encode_df['Subscription Status'].astype('float'), method=stats.pointbiserialr))
print(encode_df[['Purchase Amount USD', 'Previous Purchases', 'Age', 'Review Rating']].corrwith(encode_df['Gender_Male'].astype('float'), method=stats.pointbiserialr))
print(encode_df[['Purchase Amount USD', 'Previous Purchases', 'Age', 'Review Rating']].corrwith(encode_df['Gender_Female'].astype('float'), method=stats.pointbiserialr))
print(encode_df[['Purchase Amount USD', 'Previous Purchases', 'Age', 'Review Rating']].corrwith(encode_df['Discount Applied'].astype('float'), method=stats.pointbiserialr))
print(encode_df[['Purchase Amount USD', 'Previous Purchases', 'Age', 'Review Rating']].corrwith(encode_df['Promo Code Used'].astype('float'), method=stats.pointbiserialr))
print(stats.spearmanr(encode_df['Age'], encode_df['Purchase Amount USD']).pvalue)
print(stats.spearmanr(encode_df['Age'], encode_df['Review Rating']).pvalue)
print(stats.spearmanr(encode_df['Age'], encode_df['Previous Purchases']))
print(stats.spearmanr(encode_df['Purchase Amount USD'], encode_df['Previous Purchases']))
print(stats.spearmanr(encode_df['Purchase Amount USD'], encode_df['Review Rating']))
print(stats.spearmanr(encode_df['Previous Purchases'], encode_df['Review Rating']))


# In[195]:


# now focus on categorical data :

# encode categorical data:
encoder = OrdinalEncoder()
encoded_df2 = df.copy()

encoded_df2['Gender'] = encoded_df2['Gender'].map({'Male': 1.0, 'Female': 0.0})
encoded_df2['Subscription Status'] = encoded_df2['Subscription Status'].map({'Yes': 1.0, 'No': 0.0})
encoded_df2['Discount Applied'] = encoded_df2['Discount Applied'].map({'Yes': 1.0, 'No': 0.0})
encoded_df2['Promo Code Used'] = encoded_df2['Promo Code Used'].map({'Yes': 1.0, 'No': 0.0})

for col in encoded_df2.select_dtypes(['object']).columns:
    encoded_df2[col] = encoder.fit_transform(encoded_df2[col].values.reshape(-1, 1))
    
# make a function to calculate correlation between 2 features with Cramer's V correlation :
def cramers_corr(v1, v2):
    cross_tab =np.array(pd.crosstab(v1, v2, rownames=None, colnames=None))
    x2 = stats.chi2_contingency(cross_tab)[0]
    n = np.sum(cross_tab)
    min_dimen = min(cross_tab.shape)-1
    cramer = np.sqrt((x2/n) / min_dimen)
    return cramer

# calculate cramer's correlation across all columns and rows :
rows= []
for v1 in encoded_df2:
    col = []
    for v2 in encoded_df2:
        cramers =cramers_corr(encoded_df2[v1], encoded_df2[v2])
        col.append(round(cramers,3)) 
    rows.append(col)
    
# put result into data frame :
cramer_results = np.array(rows)
cramer_df = pd.DataFrame(cramer_results, columns = encoded_df2.columns, index =encoded_df2.columns)
print(cramer_df)

# heatmap to visualize it :
plt.subplots(figsize=(12,12))
cramer_corr = cramer_df.corr()
sns.heatmap(cramer_corr, cmap="icefire", vmin=0, vmax=1.0, fmt='.3f', annot=True)
plt.show()

# p value:
cramer_pvalue = []
cramer = []
for v1 in cramer_df:
    col = []
    for v2 in cramer_df:
        cramers, pvalue =stats.pearsonr(cramer_df[v1], cramer_df[v2])
        col.append(pvalue) 
        cramer.append(cramers)
    cramer_pvalue.append(col)

cramer_pvalues = np.array(cramer_pvalue)
cramer_pvalue_df = pd.DataFrame(cramer_pvalues, columns = encoded_df2.columns, index =encoded_df2.columns)
print(cramer_pvalue_df)


# In[196]:


# Visualisations of Data :
# Note : Only 'Males' have a 'Subscription status'
import pyarrow
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe_connected'
pl_data = pl.read_csv('shopping_trends_updated.csv')
filter = pl_data.group_by(by=("Location", "Season", "Category", "Color")).agg([pl.count("Item Purchased").alias("count")])
 
pd_df = filter.to_pandas()
print(filter.sort("count"))

fig = px.sunburst(pd_df,
                  path=["Location", "Season", "Color"],
                  values="count",
                  width=750, height=750,
                  color_continuous_scale="BrBG",
                  color="count",
                  title="Number of purchases grouped by Location,Season,Color"
                  )
fig.show()


# In[197]:


filter = pl_data.group_by(by=("Season", "Color")).agg([pl.count("Color").alias("count")])
pd_df = filter.sort("count", descending=True)
pd_df = pd_df.to_pandas()
print(pd_df.head(15))
fig = px.sunburst(pd_df,
                  path=["Season", "Color"],
                  values="count",
                  width=750, height=750,
                  color="count",
                  title="Number of purchases grouped by Season,Color"
                  )
fig.show()


# In[198]:


filter = pl_data.group_by(by=("Location", "Color")).agg([pl.count("Color").alias("count")])
pd_df = filter.sort("count", descending=True)
pd_df = pd_df.to_pandas()
print(pd_df.head(15))
fig = px.treemap(pd_df,
                  path=["Location", "Color"],
                  values="count",
                  width=750, height=750,
                  color_continuous_scale="purple",
                  color="count",
                  title="Number of purchases grouped by Location,Color"
                  )
fig.show()


# In[199]:


filter = pl_data.group_by(by=("Item Purchased", "Gender")).agg([pl.count("Gender").alias("count")])
pd_df = filter.sort("count", descending=True)
pd_df = pd_df.to_pandas()
print(pd_df.head(10))
print(pd_df[pd_df["Gender"]=="Female"].head(10))
fig = px.treemap(pd_df,
                  path=["Item Purchased", "Gender"],
                  values="count",
                  width=750, height=750,
                  color_continuous_scale="purple",
                  color="count",
                  title="Number of purchases grouped by Item Purchased,Gender"
                  )
fig.show()


# In[200]:


#
filter = pl_data.group_by(by=("Season", "Item Purchased")).agg([pl.count("Item Purchased").alias("count")])
pd_df = filter.sort("count", descending=True)
pd_df = pd_df.to_pandas()
print(pd_df.head(15))
fig = px.sunburst(pd_df,
                  path=["Season", "Item Purchased"],
                  values="count",
                  width=750, height=750,
                  color_continuous_scale="purple",
                  color="count",
                  title="Number of purchases grouped by Season,Item Purchased"
                  )
fig.show()


# In[201]:


filter = pl_data.group_by(by=("Color", "Gender")).agg([pl.count("Item Purchased").alias("count")])
pd_df = filter.sort("count", descending=True)
pd_df = pd_df.to_pandas()
print(pd_df.head(10))
print(pd_df[pd_df["Gender"]=="Female"].head(10))
fig = px.treemap(pd_df,
                  path=["Color", "Gender"],
                  values="count",
                  width=750, height=750,
                  color_continuous_scale="purple",
                  color="count",
                  title="Number of purchases grouped by Color,Gender"
                  )
fig.show()


# In[202]:


filter = pl_data.group_by(by=("Subscription Status", "Gender", "Payment Method")).agg([pl.mean("Age").alias("mean")])
print(filter)
pd_df = filter.to_pandas()
fig = px.sunburst(pd_df,
                  path=["Subscription Status", "Gender", "Payment Method"],
                  width=750, height=750,
                  color_continuous_scale="BrBG",
                  color="mean",
                  title="Age distribution grouped by Subscription Status,Gender,Payment Method"
                  )
fig.show()


# In[203]:


filter = pl_data.group_by(by=("Subscription Status", "Discount Applied")).agg([pl.mean("Purchase Amount (USD)").alias("mean")])
print(filter)
pd_df = filter.to_pandas()
fig = px.sunburst(pd_df,
                  path=["Subscription Status", "Discount Applied"],
                  width=750, height=750,
                  color_continuous_scale="BrBG",
                  color="mean",
                  title="Purchase Amount (USD) distribution grouped by Subscription Status,Discount Applied"
                  )
fig.show()


# In[204]:


filter = pl_data.group_by(by=("Subscription Status", "Gender", "Payment Method")).agg([pl.count("Item Purchased").alias("count")])
pd_df = filter.sort("count", descending=True)
pd_df = pd_df.to_pandas()
print(pd_df.head(15))
fig = px.sunburst(pd_df,
                  path=["Subscription Status", "Gender", "Payment Method"],
                  values="count",
                  width=750, height=750,
                  color_continuous_scale="dark",
                  color="count",
                  title="Number of every Payment Method grouped by Subscription Status,Gender"
                  )
fig.show()


# In[205]:


filter = pl_data.group_by(by=("Location", "Subscription Status")).agg([pl.count("Subscription Status").alias("count")])
pd_df = filter.to_pandas()
print(filter)
fig = px.treemap(pd_df,
                  path=["Location", "Subscription Status"],
                  values="count",
                  width=750, height=750,
                  color_continuous_scale="purple",
                  color="count",
                  title="Subscription Status grouped by Location"
                  )
fig.show()


# In[206]:


filter3 = pl_data.group_by(by=("Location")).agg([pl.mean("Purchase Amount (USD)").alias("mean")])
pd_df = filter3.to_pandas()
fig = px.sunburst(pd_df,
                  path=["Location"],
                  width=750, height=750,
                  color_continuous_scale="BrBG",
                  color="mean",
                  title="Purchase Amount (USD) distribution grouped by Location"
                  )
fig.show()


# In[207]:


filter3 = pl_data.group_by(by=("Gender", "Frequency of Purchases")).agg([pl.mean("Age").alias("mean")])
pd_df = filter3.to_pandas()
fig = px.sunburst(pd_df,
                  path=["Gender", "Frequency of Purchases"],
                  width=750, height=750,
                  color_continuous_scale="BrBG",
                  color="mean",
                  title="Age distribution grouped by Gender,Frequency of Purchases"
                  )
fig.show()


# In[208]:


filter3 = pl_data.group_by(by=("Frequency of Purchases", "Gender")).agg([pl.mean("Purchase Amount (USD)").alias("mean")])
pd_df = filter3.to_pandas()
fig = px.sunburst(pd_df,
                  path=["Frequency of Purchases", "Gender"],
                  width=750, height=750,
                  color_continuous_scale="BrBG",
                  color="mean",
                  title="Purchase Amount (USD) distribution grouped by Frequency of Purchases,Gender"
                  )
fig.show()


# In[209]:


#
filter4 = pl_data.group_by(by=("Season", "Location")).agg([pl.mean("Purchase Amount (USD)").alias("mean")])
pd_df = filter4.to_pandas()
fig = px.sunburst(pd_df,
                  path=["Season", "Location"],
                  width=750, height=750,
                  color_continuous_scale="BrBG",
                  color="mean",
                  title="Purchase Amount (USD) distribution grouped by Season,Location"
                  )
fig.show()


# In[210]:


#
filter4 = pl_data.group_by(by=("Season", "Color")).agg([pl.mean("Purchase Amount (USD)").alias("mean")])
pd_df = filter4.to_pandas()
fig = px.sunburst(pd_df,
                  path=["Season", "Color"],
                  width=750, height=750,
                  color_continuous_scale="BrBG",
                  color="mean",
                  title="Purchase Amount (USD) distribution grouped by Season,Color"
                  )
fig.show()


# In[211]:


# PCA and MCA to find the most intresting patterns in data :

import prince

dff = df.copy()
dff_mca = df.copy()

cols_to_mca = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases']

# MCA model :
mca = prince.MCA(n_components = 10)

mca1 = mca.fit(dff_mca[cols_to_mca])
print(mca1.eigenvalues_summary)
mca1.plot(dff_mca[cols_to_mca])
plt.show()


# In[212]:


# standard scaler for PCA --continuous data:
scaler = StandardScaler()
cols_to_scale = ['Purchase Amount USD', 'Previous Purchases', 'Age', 'Review Rating']
# create and fit scaler, scale data :
scaler.fit(dff[cols_to_scale])
dff[cols_to_scale] = scaler.transform(dff[cols_to_scale])

# PCA model :
pca = prince.PCA(n_components = 3)

pca1 = pca.fit(dff[cols_to_scale])
print(pca.eigenvalues_summary)
pca1.plot(dff[cols_to_scale])
plt.show()


# In[213]:


# additional analysis :
# Purchase amount (USD) and Review Rating
sns.lineplot(x="Review Rating", y="Purchase Amount USD", data=df)
plt.show()


# In[214]:


sns.jointplot(x="Review Rating", y="Purchase Amount USD", data=df, kind="hex")
plt.show()


# In[215]:


# K-Means Clustering :
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
df_age_rev = df[['Age', 'Purchase Amount USD', 'Review Rating']]
print(df_age_rev)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10)).fit(df_age_rev)
visualizer.show()


# In[216]:


# Kmeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(df_age_rev)

sns.scatterplot(data=df_age_rev, x="Age", y="Purchase Amount USD", hue=kmeans.labels_)

plt.show()


# In[217]:


# kmeans
scaler = StandardScaler()
dd = scaler.fit_transform(df_age_rev)
dd = pd.DataFrame(dd, columns=['Age', 'Purchase Amount USD', 'Review Rating'])
print(dd)
kmeans3d = KMeans(n_clusters = 4, init = 'k-means++',  random_state=42)
y = kmeans3d.fit_predict(df_age_rev)
df_age_rev['cluster'] = y

color_list = ['deeppink', 'blue', 'red', 'orange', 'darkviolet', 'brown']
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

for i in range(df_age_rev.cluster.nunique()):
    label = "cluster=" + str(i+1)
    ax.scatter3D(df_age_rev[df_age_rev.cluster==i]['Review Rating'], df_age_rev[df_age_rev.cluster==i]['Purchase Amount USD'], df_age_rev[df_age_rev.cluster==i]['Age'], c=color_list[i], label=label)

ax.set_xlabel('Review Rating')
ax.set_ylabel('Purchase Amount USD')
ax.set_zlabel('Age')
plt.legend()
plt.title("Kmeans Clustering Of Features")
plt.show()


# In[218]:


ddd = encoded_df2[["Location", "Color", "Purchase Amount USD"]]
from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components = 4)
model.fit(ddd)
y = model.predict(ddd)

encoded_df2['cluster'] = y

color_list = ['deeppink', 'blue', 'red', 'orange', 'darkviolet', 'brown']
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

for i in range(encoded_df2.cluster.nunique()):
    label = "cluster=" + str(i+1)
    ax.scatter3D(encoded_df2[encoded_df2.cluster==i]["Location"], encoded_df2[encoded_df2.cluster==i]['Purchase Amount USD'], encoded_df2[encoded_df2.cluster==i]["Color"], label=label)

ax.set_xlabel("Location")
ax.set_ylabel('Purchase Amount USD')
ax.set_zlabel("Color")
plt.legend()
plt.title("GaussianMixture Clustering Of Features")
plt.show()


# In[ ]:




