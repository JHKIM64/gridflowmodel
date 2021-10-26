## model grid target location
## (37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1.png),(37.4.png,126.9),(37.4.png,127.0),(37.4.png,127.1.png)


import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gridflowpack.CNSEW as CNSEW
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from tensorflow import keras
import onegridregressionmodel as ogr_model

location ={(37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)}
def get_trained_model() :
    num_model=0
    reconstructed_model=[]
    for lat,lon in location :
        model = keras.models.load_model(str(lat) + "_" + str(lon) + "_model")
        #model = ogr_model.onegridmodel(lat,lon)
        reconstructed_model.append(model)
        num_model += 1

    print(num_model)

get_trained_model()