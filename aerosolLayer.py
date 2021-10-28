import sys

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as amp
from numba import njit, cuda
from timeit import default_timer as timer
import cartopy.crs as ccrs

aerosol = xr.open_dataset("/home/intern01/jhk/Observation/EA_AQ_1920.nc")
weather = xr.open_dataset("/home/intern01/jhk/ECMWF_Land/ECMWR_EA_CLCOND_1920.nc")


index = aerosol.indexes
time = np.array(index.__getitem__('time'))
lon =  np.array(index.__getitem__('longitude'))
lat =  np.array(index.__getitem__('latitude'))

wlon =  np.array(weather.indexes.__getitem__('longitude'))
wlat =  np.array(weather.indexes.__getitem__('latitude'))

X,Y = np.meshgrid(lon,lat)
U = np.array(weather.u10.isel(time=0).values)
V = np.array(weather.v10.isel(time=0).values)
wX, wY = np.meshgrid(wlon,wlat)
C = np.array(aerosol.PM25.isel(time=0).values)

fig = plt.figure(figsize =(9,6))
box = [112, 114, 28, 28.5]
scale = '50m'
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(box,crs=ccrs.PlateCarree())
ax.coastlines(scale)
ax.set_xticks(np.arange(box[0], box[1],0.1), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(box[2], box[3],0.1), crs=ccrs.PlateCarree())
ax.grid(b=True)


c = ax.pcolormesh(X,Y,C,cmap="gist_ncar")
Q=ax.quiver(wX,wY,U,V,scale=40, scale_units='inches')

start = timer()
print(aerosol.sel(latitude=38.8, longitude=115.4))

def update_colormesh(t):
    U = np.array(weather.u10.isel(time=t).values)
    V = np.array(weather.v10.isel(time=t).values)
    C = np.array(aerosol.PM25.isel(time=t).values)

    c.set_array(C)
    Q.set_UVC(U, V)
    ax.set_title("Time="+np.datetime_as_string(aerosol.time[t].values, unit="h"))
    print(t)
    return Q,c,

anim = amp.FuncAnimation(fig, update_colormesh, frames=20,interval=300)
anim.save('china.gif')
print(timer()-start)
plt.show()
# def windplot(t) :
#     initFrame.set_array(weather.isel(time=t).values.flatten())
#     # weather.isel(time=t).plot.quiver(x='longitude',y='latitude',u='u10',v='v10')
#
# initFrame = weather.isel(time=0).plot.quiver(x='longitude',y='latitude',u='u10',v='v10')
#
# ani = amp.FuncAnimation(fig,windplot,frames=20)
# plt.show()
#
# for t in range(0,20) :
#     windplot(t)
#     plt.show()



# convert -delay value(35-40) -loop 0 'filename image_*.png' 'test.gif'