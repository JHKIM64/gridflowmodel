## near seoul
## 0.1*0.1 grid 8개
## boundary condition grid 11개

# method
## 8개 grid별 모델학습(동서남북중앙 값 시계열 변화 학습) 이후 앙상블학습 (bagging)을 통해 모델 학습
## data overfitting예방

## model grid target location
## (37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)

import pandas as pd
import dask.dataframe as dd
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gridflowpack.CNSEW as CNSEW
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score

df = CNSEW.df_CNSEW(37.6,126.9)
df.dropna(axis=0, inplace=True)
print(df.isnull().sum())
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

plt.show()

######## split into input x and output y
train_x = train_df.drop(columns='Nt_CPM25',axis=1)
train_y = train_df.Nt_CPM25
print(train_x, train_y)

val_x = val_df.drop(columns='Nt_CPM25',axis=1)
val_y = val_df.Nt_CPM25

# create model
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = Sequential()
    model.add(Dense(20, activation="relu", input_dim=55, kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))

# Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model
    model.fit(train_x, train_y, epochs=100, batch_size=5,  verbose=1)

# Calculate predictions
    PredTestSet = model.predict(train_x)
    PredValSet = model.predict(val_x)

# Save predictions
    np.savetxt("trainresults.csv", PredTestSet, delimiter=",")
    np.savetxt("valresults.csv", PredValSet, delimiter=",")

#Plot actual vs predition for training set
TestResults = np.genfromtxt("trainresults.csv", delimiter=",")
plt.plot(train_y,TestResults,'ro')
plt.title('Training Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
#Compute R-Square value for training set
TestR2Value = r2_score(train_y,TestResults)
print("Training Set R-Square=", TestR2Value)

#Plot actual vs predition for validation set
ValResults = np.genfromtxt("valresults.csv", delimiter=",")
plt.plot(val_y,ValResults,'ro')
plt.title('Validation Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.show()
#Compute R-Square value for validation set
ValR2Value = r2_score(val_y,ValResults)
print("Validation Set R-Square=",ValR2Value)

