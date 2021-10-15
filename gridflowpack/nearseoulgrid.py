import xarray as xr
import numpy as np
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

def to_xarray(dic) :
    for loc in dic :
        ns_aq_data = aerosol.sel(loc)

print(ns_aq_data)