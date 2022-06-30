## Introduction



Companies are collecting tons of data, and with this, the need for processed data is highly increasing. In the [previous blogs](https://www.enjoyalgorithms.com/blog/data-pre-processing-for-machine-learning/), we have seen the following data preprocessing techniques in theory:

* Feature Selection
* Feature Quality Assessment
* Feature Sampling
* Feature Reduction

In this blog, we will do hands-on on all these preprocessing techniques. We will use different datasets for demonstration and briefly discuss the intuition behind the methods. So let's start without any further delay.

## **Feature Selection**

Selecting the appropriate feature for data analysis requires the knowledge of feature selection methods and some domain knowledge. Combining both allows us to choose the optimum features necessary for tasks. We will discuss one technique from each feature selection method. Let's first see the types of feature selection methods:

### **Wrapper Methods**

The wrapper method requires a machine-learning algorithm to find the optimum features. This method generally has a high computational time. There are many methods for finding the significant features, but we will confine ourselves to **Sequential Feature Selector** (for step-wise selection). This is a greedy algorithm that sequentially checks the model's performance with each feature and sequentially adds to the features.

```python

```

![Sequential Feature Selector output](https://cdn-images-1.medium.com/max/1600/1*D9rHRr1Hedtl2Y8XoFvIIQ.png "Sequential Feature Selector output")

### **Filter Methods**

Filter methods are a set of statistical techniques required for measuring the importance of features in a dataset. It includes Correlation, Chi-Square test, information gain, etc. These methods are comparatively faster in terms of time complexity. Let's implement correlation as a feature selection criteria:

```python

```

![correlation analysis](https://cdn-images-1.medium.com/max/1600/1*bNkuvdSumS3uDMmUmoIRtw.png "correlation analysis")

RAD and TAX are highly correlated, and therefore, we can remove one with a low value of absolute correlation with the target variable (MEDV). "RAD" can be removed. Also, we can set a threshold value of the absolute correlation to remove the less informative features. For instance, if we select 0.40 correlation as a feature selection criteria for the modeling, we are left with 'INDUS,' 'NOX,' 'RM,' 'TAX,' 'PTRATIO,' 'LSTAT.' as the final set of features.  

### **Embedded Methods**

Embedded methods combine the qualities of Filter and Wrapper methods. It works by iteratively building the models while also evaluating their performance. Based on performance, the best set of parameters is chosen. Again, there are many techniques within embedded methods, but we will use Random Forest Feature Importance. Let's implement RF Feature Importance for selecting the optimum features:

```python

```

![Feature importance of random forest](https://cdn-images-1.medium.com/max/1600/1*G1UJA0xGb7FbFbvoyXpCdw.png "Feature importance of random forest")

From the above plot, we can conclude 'RM,' 'LSTAT,' 'DIS,' 'CRIM,' and 'NOX' as the optimal features in the analysis. Embedded methods fall between the filter and wrapper methods in time complexity. 

We could have discussed more techniques from each method, but we will revisit this topic in a separate blog on Feature Selection techniques. Let's move on to the feature quality assessment.

## Feature Quality Assessment

This section deals with the quality of data. Data is prone to noise, human errors, scaling errors, and reading errors. Such errors induce inconsistency in the collected data. It is recommended to process the data to remove these inconsistencies, and we will do the same on the Advanced Regression house price dataset. 

### **Missing Values**

Missing values are inevitable in raw data. Sometimes, sensors don't work the way we want, or it could be a case where data isn't available, and such cases result in the presence of missing values in the data. Let's look at ways of treating the missing values in the data.

#### **Row Elimination**

Sometimes, only a few rows have missing values. Removing 1 to 10% of data doesn't make significant differences; however, in time-series data where sequential data points are collected over a fixed period, this method can cause problems. 

#### **Imputation**

In this method, we replace the missing values with specific imputed values. The imputation methods are different for numerical and categorical features. It is recommended to separate the numeric and categorical features beforehand. Afterward, we will break the imputation problem as the imputation of categorical features and numerical features. 

We are going to apply some imputation techniques to the given dataset. Take a look at the dataset:

```python

```

![Imputation data snippet missing data](https://cdn-images-1.medium.com/max/1440/1*8Xq2EanUxYomqLXTsk40TQ.jpeg "Imputation data snippet missing data")

**Imputation methods for numerical variables:**

* Mean/Median Imputation

```python

```

![Mean imputation on price data](https://cdn-images-1.medium.com/max/1440/1*gP7E9G7CUQAUH7tm6ISZGg.jpeg "Mean imputation on price data")

```python

```

![median imputation](https://cdn-images-1.medium.com/max/1440/1*VGK6fv5Asy-HenKhZWN5cQ.jpeg "median imputation")

* Forward Fill/Backward Fill

```python

```

![forward fill figure](https://cdn-images-1.medium.com/max/1440/1*jU7I0MzfwPyt33vhQnu2iQ.jpeg "forward fill figure")

```python

```

![backward fill figure](https://cdn-images-1.medium.com/max/1440/1*KU1qdYQlLwuFiwDUwvkJzg.jpeg "backward fill figure")

* Interpolation

```python

```

![Linear interpolation output](https://cdn-images-1.medium.com/max/1440/1*BJ1v0z6dYbBWmhEOnOQumQ.jpeg "Linear interpolation output")

**Imputation methods for categorical variables:**

* Frequent Category Imputation

```python

```

![Frequent category imputation](https://cdn-images-1.medium.com/max/1440/1*z7OOAZvynvx3lBuXPpz-hw.jpeg "Frequent category imputation")

* Adding a new category as "Missing."

```python

```

![missing class added ](https://cdn-images-1.medium.com/max/1440/1*9dhTADMEjg4ogZFB03SpEg.jpeg "missing class added ")

### Outlier

Outliers are the abnormal readings or data points present in the dataset. They are highly distinguishable and need to be removed since they can cause severe problems during the model building phase. There are two ways of detecting an outlier from the dataset:

* Using Statistical Method
* Using Machine Learning Algorithms

Let's look at some statistical methods available for the removal of outliers from the data:

#### **Inter Quartile Range (IQR)**

If you are unfamiliar with Quartile, please have a [look at this](https://www.investopedia.com/terms/q/quartile.asp). Inter Quartile Range is nothing but the difference between the third and first Quartile. More formally:

> IQR = Q3 - Q1

Data points above Q3 + 1.5 IQR and below Q1 – 1.5IQR are treated as outliers. Let's implement this in our dataset:

```python

```

![Inter Quartile Range (IQR) data](https://cdn-images-1.medium.com/max/1440/1*BHDmaaFYSApWbqeLNfKKhA.jpeg "Inter Quartile Range (IQR) data")

```python

```

![Inter Quartile Range (IQR) output](https://cdn-images-1.medium.com/max/1600/1*gvfKMP7gk0o5axZveFJ1FA.png "Inter Quartile Range (IQR) output")

#### **Standard Deviation**

This approach is relatively simple and effective as it only requires the computation of mean and standard deviation to calculate the upper and lower bounds. Any value within these bounds is a safe value and otherwise anomaly. Let's implement this approach:

```python

```

![standard deviation output](https://cdn-images-1.medium.com/max/1600/1*9-U3kcfGb_9aBH5Vg2nDsA.png "standard deviation output")

### **Duplicate Values**

Duplicate values can be present due to various factors. Such data points can lead to model bias. Let's remove the duplicate rows from the dataset:

```python

```

## Data Sampling

Data Sampling is a statistical technique used to manipulate and examine a representative subset of the original dataset. It helps identify and visualize trends and patterns while also reducing computational efforts. Following are some frequently used data sampling methods:

### Upsampling/Downsampling

Upsampling and Downsampling techniques are generally used in Classification problems to address the class imbalance. It simply works by reducing or increasing the samples from the majority or minority classes depending on the type of Sampling used. Let's apply them one by one:

Consider a binary classification with unbalanced labels (Class_1 has more samples than Class_2). We aim to downsample the data points for Class_1 to eliminate the class imbalance.

**Downsampling:**

```python

```

![Downsampling](https://cdn-images-1.medium.com/max/1600/1*ErtQhnZZapntNeRXpBovbQ.png "Downsampling")

After Downsampling

**Upsampling:**

```python

```

![Upsampling](https://cdn-images-1.medium.com/max/1600/1*qzBwGE9PnoXX_wap0MDTCw.png "Upsampling")

After Upsampling

### **Stratified Sampling**

Stratified Sampling is the most commonly used sampling technique in classification problems to make the test and training dataset. This method samples the data such that classes are divided into homogenous sub-groups, maintaining the same proportions. These homogenous sub-groups are also known as strata.

![Stratified sampling](https://cdn-images-1.medium.com/max/1600/1*yiN6zTUA4pDwEXAbeapJOA.png "Stratified sampling")

Source: Data Science Made Simple

```python

```

![Data snippet](https://cdn-images-1.medium.com/max/1440/1*Q_CihNdCSlu3gzL5cRr9Dw.jpeg "Data snippet")

Let's select the stratified samples of the above dataset based on the dress color code. As we can see, 50% of participants are wearing Black, 30% of participants are wearing Green, and the rest 20% of participants are wearing Red dresses. Our stratified sampled dataset must retain this proportion. Let's implement stratified Sampling:

```python

```

![stratified sampling output](https://cdn-images-1.medium.com/max/1440/1*EgW4iLJzhL02EaVIZs1vcQ.jpeg "stratified sampling output")

### Simple Random Sampling

As the name suggests, it is a statistical sampling method where each data point in a dataset has an equal probability of being selected in a dataset. This sampling results in an unbiased representation of a group. Let's apply random SamplingSampling in the dress color code dataset:

```python

```

![Random sampling output](https://cdn-images-1.medium.com/max/1440/1*wH98S_EbVFKw2PujTrA9uw.jpeg "Random sampling output")

## Feature Reduction

Nowadays, firms collect a lot of features for the analysis but using all of them for modeling is not feasible since this will come at a computational cost, and dimensionality reduction addresses this problem by reducing the number of features and keeping them intact with the same level of information. 

Following are some techniques available for reducing the dimensionality of the dataset. Let's discuss them along with their application:

### Principal Component Analysis (PCA)

PCA is a statistical dimensionality reduction method used to reduce the number of dimensions from a larger dataset. This feature reduction technique comes at the cost of minimal information loss, but this can save a lot of computational efforts and make the visualization better. Machine learning algorithms are much easier to implement with reduced features. Let's implement PCA to the dataset. 

```python

```

![Data snippet](https://cdn-images-1.medium.com/max/1600/1*43f44niU9rQjcHjxWrznYg.png "Data snippet")

Transactional Data

Generally, we first normalize the dataset and then apply PCA, but features are already scaled in our case, so normalization is not required.

```python

```

![features and their weightage](https://cdn-images-1.medium.com/max/1600/1*kx2eio9ebWAPDavfW_andg.png "features and their weightage")

Percentage Variance captured by each feature.

The first 17 Principal Components cover 85% information (variance) of the original dataset. Let's reduce the dataset to 17 features:

```python

```

Initially, we had 28 features, and we reduced the number of features to 17 at the cost of 15% information loss. 

### t-SNE

t-Distributed Stochastic Neighbor Embedding (t-SNE) is another dimensionality reduction technique. It is specifically used for the visualization of high-dimensional data. Moreover, It is computationally expensive, and hence, there are some severe limitations to this technique. For instance, using a "**Second dimensionality reduction technique**" is highly advisable in some cases of very high dimensional data. Let's apply t-SNE to our dataset:

```python

```

## Conclusion

In this article, we have demonstrated the methodology of data preprocessing techniques. We looked at different feature selection methods and applied them to the Boston house price dataset. Further, we looked at some feature quality assessment techniques where we discussed and used Missing value imputation techniques and Outlier Detection techniques. Moving onwards, we also observed some sampling methods and dimensionality reduction techniques and applied them to a dataset. We hope you enjoyed the article.

Enjoy Learning, Enjoy Algorithms!