import gridflowpack.CNSEW as CNSEW
import ensemblemodel as ensemble
import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# china 117.2, 31.8 // 28.2, 112.9
df_ch1 = CNSEW.df_CNSEW(31.8, 117.2)
df_ch1.dropna(axis=0, inplace=True)
df_ch2 = CNSEW.df_CNSEW(28.2, 112.9)
df_ch2.dropna(axis=0, inplace=True)

def regularization(df) :
    train_mean = df.mean()
    train_std = df.std()

    train_df = (df - train_mean) / train_std
    return train_df

model = keras.models.load_model("onegridensemble")

def run(lat,lon,step, df) :
    esbmodel, test_df = model, regularization(df)
    n = len(test_df)
    timestep = step
    timestep_df = test_df[n - timestep:]
    test_df = test_df[0:n-timestep]

    ensemble.predict_ensemble(esbmodel, test_df,lat,lon)

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

run(28.2, 112.9, 168, df_ch2)
run(31.8, 117.2, 168, df_ch1)