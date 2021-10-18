import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

aerosol = xr.open_dataset("/home/intern01/jhk/Observation/EA_AQ_1920.nc")
weather = xr.open_dataset("/home/intern01/jhk/ECMWF_Land/ECMWR_EA_CLCOND_1920.nc")

##boundary grid - outside of inner grid(+모양으로 둘러쌓음)
boundary = [(37.7,126.9),(37.7,127.0),(37.6,127.1),(37.5,127.2),(37.4,127.2),(37.3,127.1),(37.3,127.0),(37.3,126.9),(37.4,126.8),(37.5,126.8),(37.6,126.8)]
##inner grid - seoul 8 grid
inner_grid = [(37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)]

def to_dic(array) :
    dic = np.array([])
    for i in array :
        dic = np.append(dic,{'latitude':i[0],'longitude':i[1]})
    return dic

def to_xarray(dic, xr) :
    ns_aq_data = pd.DataFrame()
    for loc in dic :
        data = xr.sel(loc).expand_dims(["latitude","longitude"]).to_dataframe()
        ns_aq_data = pd.concat([ns_aq_data,data]).sort_index()

    return ns_aq_data

def gridData() :
    ae_bd = to_xarray(to_dic(boundary),aerosol)
    # print("ad_bd",ae_bd)
    ae_ig = to_xarray(to_dic(inner_grid),aerosol)
    # print("ad_ig", ae_ig)
    wt_bd = reset_Loc(to_xarray(to_dic(boundary),weather))
    wt_ig = reset_Loc(to_xarray(to_dic(inner_grid), weather))

    df_bd = pd.merge(ae_bd, wt_bd, left_index=True, right_index=True, how='left').sort_index()
    df_ig = pd.merge(ae_ig, wt_ig, left_index=True, right_index=True, how='left').sort_index()
    df_all = pd.concat([df_bd,df_ig]).sort_index()
    # print(df_all)
    return df_bd, df_ig, df_all

def reset_Loc(df) :
    df = df.reset_index(['time','latitude','longitude'])
    df.latitude = np.round(df.latitude,1)
    df.longitude = np.round(df.longitude, 1)
    df = df.set_index(['time','latitude','longitude'])
    return df

gridData()