## model grid target location
## (37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)


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
import gridflowpack.CNSEW as CNSEW
# VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

location ={(37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)}
def get_trained_model() :
    reconstructed_model=list()
    for lat,lon in location :
        model = keras.models.load_model(str(lat) + "_" + str(lon) + "_model")
        #model = ogr_model.onegridmodel(lat,lon)
        reconstructed_model.append(model)
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
    print(x_train)
    y_train = train_df.Nt_CPM25
    print(y_train)
    # ensemble 할 model 정의
    models = get_trained_model()
    ### create model using GPU
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():  ###gpu process
    # soft vote
        k_fold=KFold(n_splits=8, shuffle=True)
        soft_vote = VotingRegressor(models, n_jobs=3)
        #soft_vote_cv = cross_validate(soft_vote, x_train, y_train, cv=k_fold)
        soft_vote.fit(x_train, y_train)

    return soft_vote, test_df

def predict_ensemble(esbmodel, test_df) :
    test_x = test_df.drop(columns='Nt_CPM25', axis=1)
    test_y = test_df.Nt_CPM25

    ### create model using GPU
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():  ###gpu process
        ### prediction
        PredTestSet = esbmodel.predict(test_x)

        ### Save predictions
        np.savetxt("testresults.csv", PredTestSet, delimiter=",")

    ValResults = np.genfromtxt("valresults.csv", delimiter=",")
    plt.plot(test_y, ValResults, 'ro')
    plt.title('Test Set')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.show()

def run() :
    esbmodel, test_df = ensemble(37.6,126.9)
    n = len(test_df)
    timestep = 24
    test_df = test_df[0:n-timestep]
    timestep_df = test_df[n-timestep:]

    predict_ensemble(esbmodel, test_df)

    # for i in range(0,timestep) :
    #     strategy = tf.distribute.MirroredStrategy()
    #
    #     with strategy.scope():
    #         esbmodel.predict(test_x)
run()