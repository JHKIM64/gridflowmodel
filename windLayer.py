import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as amp
from numba import njit, cuda
from timeit import default_timer as timer

import cartopy.crs as ccrs

weather = xr.open_dataset("/home/intern01/jhk/ECMWF_Land/ECMWR_EA_CLCOND_1920.nc")

index = weather.indexes
time = np.array(index.__getitem__('time'))

lon =  np.array(index.__getitem__('longitude'))
lat =  np.array(index.__getitem__('latitude'))

X,Y = np.meshgrid(lon,lat)
U = np.array(weather.u10.isel(time=0).values)
V = np.array(weather.v10.isel(time=0).values)


fig= plt.figure(figsize =(8.8,8.5))
box = [126.3, 128, 36.5, 38]
scale = '50m'
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(box,crs=ccrs.PlateCarree())
ax.set_xticks(np.arange(box[0], box[1],0.1), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(box[2], box[3],0.1), crs=ccrs.PlateCarree())
ax.grid(b=True)
ax.coastlines(scale)
Q=ax.quiver(X,Y,U,V,scale=20, scale_units='inches')

start = timer()
def update_quiver(t):
    U = np.array(weather.u10.isel(time=t).values)
    V = np.array(weather.v10.isel(time=t).values)

    Q.set_UVC(U,V)
    ax.set_title("Time="+np.datetime_as_string(weather.time[t].values, unit="h"))
    return Q,

anim = amp.FuncAnimation(fig, update_quiver, frames=10,interval=300)
anim.save('test1.gif')
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