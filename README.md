## Project Introduction 
The data analysis project involves the design of a complete machine learning solution. In
particular, the project revolves around the task of identifying the music genre of songs. This
is useful as a way to group music into categories that can be later used for recommendation
or discovery. 

## The Data description
The data is split into two datasets: a training data set with 4363 songs, and a test set dataset
with 6544 songs. Each song has 264 features, and there are 10 possible classes in total.
The dataset is a custom subset of the Million Song Dataset, and the labels were obtained
from AllMusic.com. 

#### Music genre are as follows:

1. Pop_Rock
1. Electronic
1. Rap
1. Jazz
1. Latin
1. RnB
1. International
1. Country
1. Reggae
1. Blues 


#### Features of the song 
The features provided are a summary representation of the 3 main components of music:
timbre, pitch (melody and harmony) and rhythm. A very brief description of these
components: timre, pitch and rythm.  The final feature vector of each song consists of 264
dimensions: 168 values for the rhythm patterns (24 bands, 7 statistics), 48 values for the
chroma (12 bands, 4 statistics), and 48 values for the MFCCs (12 bands, 4 statistics).
![song_genre_data_desc](song_genre_data_desc.PNG)

## Loading the raw data 

#### Code snippet to load data 
```python
// Python pandas to load data 
# Load the data and cleanup
df_train = pd.read_csv("train_data.csv",header=None)
df_labels_train = pd.read_csv("train_labels.csv",header=None)
df_train["labels"] = df_labels_train[:]

### Checking  for NAN values in the dataset
x = df_train.isna().sum()
for x1 in x:
    if x1 != 0:
        print(x1)
df_test = pd.read_csv("test_data.csv",header=None)
```
#### Visualizing using Histogram and correlation matrix
```python
df_train['labels'].hist()
df_subset = df_train[list(range(0,167,1))]
plt.matshow(df_subset.corr())
```
![Vis_hist](hist_mlbp.PNG)
![vis_correl](correl_mlbp.PNG)

## Data Preparation
#### Filtering the columns which are not correlated with the label column
As not all the columns might be correlated with the label columns. That means the columns which do not add value to the prediction of the label are found out in the following logic through the correlation matrix. Training and checking through this filtered train was done with taking only most correlated features with chi square correlation[1] of more than absolute of the correlation statistics of 0.05 , 0.1, 0.2 but none of those increased the accuracy more than when all the fetaures were taken. It did not improve the overall accuracy so this method was not taken into consideration for final model.

```python
from sklearn.feature_selection import chi2

import numpy as np
corr_matrix = df_train.corr()
corr_with_label = corr_matrix["labels"].sort_values(ascending=False)

top_correlated_features = []
for index in corr_with_label.index:
    if abs(corr_with_label[index]) >= 0.05:
        top_correlated_features.append(index)
print(len(top_correlated_features))
print(top_correlated_features[:10])
X_filteredfeatures_train = df_train[df_train.columns[top_correlated_features[1:]]]
```
#### Removing the columns which have almost same values
There are columns where the value is same throughout the rows so the procedure implemented for this is to check if the values of 25th percentile, 50th percentile and 75th percentile are equal then those columns were eliminated.
```python 
columns_to_include = []

for i in X_train.columns:
    df_descrip = df_train[i].describe()
    if abs(df_descrip['25%'] - df_descrip['75%']) == 0 and abs(df_descrip['25%'] - df_descrip['50%']) == 0:
        continue
    columns_to_include.append(i)
print("Number of columns having almost similar values :")
print(str(len(X_train.columns) - len(columns_to_include)))
#print(columns_to_include)
X_filteredfeatures_train = df_train[df_train.columns[columns_to_include]]
df_test_filtered = df_test[df_test.columns[columns_to_include]]
#columns got eliminated
#20 columns with similar values get eliminated 
# These are not additing any value to the ML problem 
```

#### Normalize the data
While analysing the values for each feature we observed that there were values ranging from as high as 1milion to -0.002, therefore the decision to normalize the data was taken before creating the models. Normalization helped in improving the accuracy of each of the models.
```python 
from sklearn import preprocessing
df_train_new = pd.DataFrame()
df_test_new = pd.DataFrame()
# Get column names first
names = X_filteredfeatures_train.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(X_filteredfeatures_train)
df_train_new = pd.DataFrame(scaled_df, columns=names)

scaled_df = scaler.fit_transform(df_test_filtered)
df_test_new = pd.DataFrame(scaled_df, columns=names)
```

