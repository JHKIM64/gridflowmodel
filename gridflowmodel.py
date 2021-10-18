## near seoul
## 0.1*0.1 grid 16개
## boundary condition grid 16

# method
## 16개 grid별 모델학습 이후 앙상블학습 (bagging) 을 통해 모델 학습
## 시계열 데이터를 t -> t+1 데이터의 집합으로 바꿔야함

import pandas as pd
import dask.dataframe as dd
import math
import tensorflow as tf
from tensorflow import keras

import numpy as np
import gridflowpack.nearseoulgrid as nsgrid

boundary, ingrid, allData = nsgrid.gridData()

