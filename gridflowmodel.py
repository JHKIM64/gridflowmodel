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
import gridflowpack.CNSEW

