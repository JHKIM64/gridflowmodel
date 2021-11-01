## model grid target location
## (37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)
import matplotlib.text
import pandas as pd
import pydot
import graphviz
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gridflowpack.CNSEW as CNSEW
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.utils.vis_utils import plot_model
from sklearn.metrics import r2_score
from tensorflow import keras
# import onegridregressionmodel as ogr_model
import gridflowpack.CNSEW as CNSEW
# VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

location ={(37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)}
def get_trained_model() :
    reconstructed_model=list()
    model_num=0
    for lat,lon in location :
        model = keras.models.load_model(str(lat) + "_" + str(lon) + "_model")
        #model = ogr_model.onegridmodel(lat,lon)
        reconstructed_model.append(model)
        model_num += 1
    return reconstructed_model

def ensemble(lat,lon) :
    df = CNSEW.df_CNSEW(lat, lon)
    df.dropna(axis=0, inplace=True)
    n = len(df)
    ### split dataframe to training, validation, test
    train_df = df[0:int(n * 0.8)]
    test_df = df[int(n * 0.8):]

    # num_features = df.shape[1.png]

    ### regularization data(data preprocessing)
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    ###
    x_train = train_df.drop(columns='Nt_CPM25', axis=1)
    y_train = train_df.Nt_CPM25
    # ensemble 할 model 정의
    models = get_trained_model()
    ### create model using GPU
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():  ###gpu process
        inputs = keras.Input(shape=(55,))
        mods = []
        for model in models :
            mods.append(model(inputs))
        outputs = layers.average(mods)
        averagemodel = keras.Model(inputs=inputs, outputs=outputs)
        # averagemodel.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # averagemodel.fit(x_train,y_train, epochs=100, batch_size=5, verbose=1)
        plot_model(averagemodel, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        averagemodel.save("onegridensemble")

    return averagemodel, test_df

def predict_ensemble(esbmodel, test_df, lat, lon) :
    test_x = test_df.drop(columns='Nt_CPM25', axis=1)
    test_y = test_df.Nt_CPM25

    ### prediction
    PredTestSet = esbmodel.predict(test_x)

    ### Save predictions
    np.savetxt("testresults.csv", PredTestSet, delimiter=",")

    TestResults = np.genfromtxt("testresults.csv", delimiter=",")
    plt.figure(figsize=(20,15))
    plt.plot(test_y.reset_index(drop=True), 'r', label='Obsv')
    plt.plot(TestResults, 'b', label='Pred')
    plt.title('Test Set at '+str(lat)+','+str(lon))
    plt.xlabel('timestep')
    plt.legend()
    TestR2Value = r2_score(test_y, TestResults)
    font3 = {'color': 'forestgreen','size': 14}
    plt.text(0,-2,"TestSet R-Square="+str(TestR2Value),fontdict=font3)
    plt.savefig('Test_'+str(lat)+'_'+str(lon)+'.png')
    plt.show()

def run(lat,lon,step) :
    esbmodel, test_df = ensemble(lat,lon)
    n = len(test_df)
    timestep = step
    timestep_df = test_df[n - timestep:]
    test_df = test_df[0:n-timestep]

    predict_ensemble(esbmodel, test_df,lat,lon)

    timestep_x = timestep_df.drop(columns='Nt_CPM25', axis=1)
    timestep_y = timestep_df.Nt_CPM25

    df = []
    for i in range(0,timestep-1) :
        pred_next = esbmodel.predict(timestep_x[i:i+1])
        df.append([timestep_y.iloc[i],float(pred_next)])
        timestep_x.iloc[i+1].CPM25 = pred_next

    df = pd.DataFrame(df)
    df.columns = ["Obsv","Pred"]
    plt.figure(figsize=(20, 15))
    plt.plot(df.Obsv, 'r',label='Obsv')
    plt.plot(df.Pred, 'b',label='Pred')
    plt.title(str(timestep)+' hours Prediction at '+str(lat)+','+str(lon))
    plt.xlabel('timestep')
    PredR2Value = r2_score(df.Obsv, df.Pred)
    font = {'color': 'black', 'size': 14}
    plt.legend()
    plt.text(0,-1,"Prediction R-Square=" + str(round(PredR2Value,4)), fontdict=font)
    plt.savefig(str(timestep)+'h_Pred_'+str(lat)+'_'+str(lon)+'.png')
    plt.show()

for lat, lon in location :
    run(lat,lon,168)