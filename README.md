# Analysis_Shopping_Trends
This project includes data analysis of shopping trends and customer's behaviour using descriptive and inferential analysis.

Check a file "Shopping_trends_presentation.pdf" to see a presentation of this analysis or go below to see slices of that presentation with a source code and plots.

## Table of contents :
* [Introduction](#introduction-)
* [Descriptive Analysis](#descriptive-analysis-)
* [Shopping Trends](#shopping-trends-)
* [Conclusions](#conclusions-)

## Introduction :
Dataset used in this project is in file : 'shopping_trends_updated.csv'

Full code of this data analysis in :
- jupyter lab file: 'Shopping_trends.ipynb'
- python file: 'Shopping_trends.py'

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/3.jpg)

```python
print(df.info())
```
```python
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
# 0   Customer ID             3900 non-null   int64  
# 1   Age                     3900 non-null   int64  
# 2   Gender                  3900 non-null   object 
# 3   Item Purchased          3900 non-null   object 
# 4   Category                3900 non-null   object 
# 5   Purchase Amount USD     3900 non-null   int64  
# 6   Location                3900 non-null   object 
# 7   Size                    3900 non-null   object 
# 8   Color                   3900 non-null   object 
# 9   Season                  3900 non-null   object 
# 10  Review Rating           3900 non-null   float64
# 11  Subscription Status     3900 non-null   object 
# 12  Shipping Type           3900 non-null   object 
# 13  Discount Applied        3900 non-null   object 
# 14  Promo Code Used         3900 non-null   object 
# 15  Previous Purchases      3900 non-null   int64  
# 16  Payment Method          3900 non-null   object 
# 17  Frequency of Purchases  3900 non-null   object)
```

## Descriptive Analysis :

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/4.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/5.jpg)

in code :
```python
# distributions of continuous data :
pd.option_context('mode.use_inf_as_na', True)
sns.displot(df['Age'], kde=True)
sns.displot(df['Purchase Amount USD'], kde=True)
sns.displot(df['Review Rating'], kde=True)
sns.displot(df['Previous Purchases'], kde=True)
```
categorical data :
![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/6.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/7.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/8.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/10.jpg)

# The most common of categorical data :
![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/9.jpg)

in code :
```python
# numbers of categorical data :
for col in df.select_dtypes(['object']).columns:
    fig = plt.figure(figsize=(30,10))
    ax = sns.countplot(x=df[col], palette='vlag', hue=df['Gender'], order = df[col].value_counts().index)
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()
```
# Correlations using Spearman's and Cramer's V correlations :

in code :

```python
# spearman's correlation but discrete and binary data:
plt.subplots(figsize=(12,12))
corr_s = encode_df.corr('spearman', numeric_only=True)
sns.heatmap(corr_s, cmap="rainbow", vmin=-0.9, vmax=0.9, fmt='.3f', annot=True)
plt.show()
```

```python
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
```

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/11.jpg)

## Shopping Trends :

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/12.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/13.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/14.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/15.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/16.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/17.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/18.jpg)

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/19.jpg)

in code for 1 example :

```python
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
```

# MCA for dimension reduction and patterns in categorical data,
# PCA for dimension reduction and patterns in discrete data :

in code MCA :

```python
import prince

dff = df.copy()
dff_mca = df.copy()

cols_to_mca = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases']

# MCA model :
mca = prince.MCA(n_components = 10)

mca1 = mca.fit(dff_mca[cols_to_mca])
print(mca1.eigenvalues_summary)
mca1.plot(dff_mca[cols_to_mca])
```

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/20.jpg)


in code PCA :

```python
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
```

# Plots for correlation between "Purchase Amount USD" and "Review Rating" :
Looking at PCA plot and pvalue= ~0.05782, features "Purchase Amount USD" and "Review Rating" seem to be significantly correlated 
In plots below, the Purchase Amount slightly increases as the Review Rating increases :

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/21.jpg)

in code :

```python
sns.lineplot(x="Review Rating", y="Purchase Amount USD", data=df)
sns.jointplot(x="Review Rating", y="Purchase Amount USD", data=df, kind="hex")
```

# Example od customer segmentation with K-Means Clustering :

![Picture](https://github.com/claudia13062013/Analysis_Shopping_Trends/blob/main/visual_plots/slides_presentation/22.jpg)

in code :

```python
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
```

## Conclusions :
This data analysis helped in better understanding of shopping trends like

-- prefered colors,categories or sizes of clothes

-- changes in trends due to gender, age, location

-- customer segmentation based on shopping trends


Thank you for reading !
