## Prediction and Classification of Songs in Genres 
The data analysis project involves the design of a complete machine learning solution. In
particular, the project revolves around the task of identifying the music genre of songs. This
is useful as a way to group music into categories that can be later used for recommendation
or discovery. 

## The Data description
The data is split into two datasets: a training data set with 4363 songs, and a test set dataset
with 6544 songs. Each song has 264 features, and there are 10 possible classes in total.
The dataset is a custom subset of the Million Song Dataset, and the labels were obtained
from AllMusic.com. 

### Music genre are as follows:

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


### Features of the song 
The features provided are a summary representation of the 3 main components of music:
timbre, pitch (melody and harmony) and rhythm. A very brief description of these
components: timre, pitch and rythm.  The final feature vector of each song consists of 264
dimensions: 168 values for the rhythm patterns (24 bands, 7 statistics), 48 values for the
chroma (12 bands, 4 statistics), and 48 values for the MFCCs (12 bands, 4 statistics).
![song_genre_data_desc](song_genre_data_desc.PNG)

### Loading the dataset and creating the Train and Test datasets
```python
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
# Taking only the features in X_train
X_train = df_train[list(range(264))]
# Assign labels to y_train
y_train = df_train['labels']
# filtered features initialized to X_train
# We will assign filtered features to the X_filteredfeatures_train
X_filteredfeatures_train = X_train
# Printing the frequency of each class
df_train['labels'].value_counts()
```
