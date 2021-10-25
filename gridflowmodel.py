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
import xarray as xr
from tensorflow import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gridflowpack.CNSEW as CNSEW
import tensorflow.keras as keras

df = CNSEW.df_CNSEW(37.6,126.9)

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