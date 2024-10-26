#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Libraries 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


ds=  xr.open_dataset(r"G:/IMDAA_Regrid_1.08_1990_2022/IMDAA_merged_1.08_1990_2020.nc")
ds


# In[3]:


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# In[10]:


import cartopy.feature as cfeature
import cartopy.crs as ccrs


# In[11]:


mean_all = ds.mean(dim='time')
z500 = mean_all
data_values = z500['APCP_sfc'].values

# Get the extent of your xarray dataset
lon_min, lon_max, lat_min, lat_max = z500.lon.min(), z500.lon.max(), z500.lat.min(), z500.lat.max()

# Create a basic plot within the extent of your xarray data using cartopy
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
im = ax.imshow(data_values, cmap='jet', origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], transform=ccrs.PlateCarree())

ax.tick_params( labelcolor='black', labelsize='large', width=2)
# Add country borders and coastlines within the extent
ax.add_feature(cfeature.COASTLINE, linewidth=1)
#ax.add_feature(cfeature.BORDERS, linewidth=1)
#world.boundary.plot(ax=ax, linewidth=1, color='black')

# Add Cartopy graticules (latitude and longitude lines)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
#ax.set_xticks(range(int(lon_min), int(lon_max) + 1, 10), crs=ccrs.PlateCarree())
#ax.set_yticks(range(int(lat_min), int(lat_max) + 1, 10), crs=ccrs.PlateCarree())
# plt.xticks(fontweight='bold')
# plt.yticks(fontweight='bold')
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('(Kg/m^2)', fontsize = 12, weight="bold")
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), weight="bold")
plt.title('Mean TP6h ' , fontweight='bold')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.grid(True)
plt.savefig('TP6h.png', dpi = 600)
plt.show()



# # Evaluation Metrics

# In[3]:


def compute_rmse(prediction, actual,  mean_dims = ('time', 'latitude', 'longitude')):
  error = prediction - actual
  rmse = np.sqrt(((error)**2 ).mean(mean_dims))
  return rmse


# In[4]:


def compute_mae(prediction, actual, mean_dims = ('time', 'latitude', 'longitude')):
    error = prediction - actual
    mae = np.abs(error).mean(mean_dims)
    return mae


# In[5]:


def compute_acc(prediction, actual):
    clim = actual.mean('time')
    try:
        t = np.intersect1d(prediction.time, actual.time)
        pred_anomaly = prediction.sel(time=t) - clim
    except AttributeError:
        t = actual.time.values
        pred_anomaly = prediction - clim
    act_anomaly = actual.sel(time=t) - clim
    
    pred_norm = pred_anomaly - pred_anomaly.mean()
    act_norm = act_anomaly - act_anomaly.mean()

    acc = (
            np.sum(pred_norm * act_norm) /
            np.sqrt(
                np.sum(pred_norm ** 2) * np.sum(act_norm ** 2)
            )
    )
    return acc


# # Basic Baseline Models

# In[6]:


# training dataset selection
train_years = slice('1990', '2017')
# validation dataset selection (this dataset helps with overfitting)
valid_years = slice('2018', '2018')
# test dataset selection
test_years = slice('2019', '2020')


# In[7]:


var_name = ['HGT_prl', 'TMP_prl', 'TMP_2m', 'APCP_sfc'] # [H500, T850, T2m, TP6h]


# ## Persistence

# In[71]:


# Each day the data has four observations at 00 UTC, 06 UTC, 12 UTC and 18 UTC
lead_time_steps = np.arange(4, 64, 4) 
lead_time_steps


# In[72]:


# Compute the rmse for each lead_time_steps

df_error = pd.DataFrame()
for var in var_name:
    error_rmse = []
    error_mae = []
    error_acc = []
    for i,j in enumerate(lead_time_steps):

 # compute persistent forecast
        persistence_fc = ds.sel(time=test_years).isel(time=slice(0, -j))
        persistence_fc['time'] = persistence_fc.time + np.timedelta64(i+1, 'D').astype('timedelta64[ns]')
        target = ds.sel(time=test_years)
        target = target.isel(time=slice(j, None))
        # change the variable name according to the dataset
        error = compute_rmse(persistence_fc, target)[var].values.item() 
        error_rmse.append(error)
        error = compute_mae(persistence_fc, target)[var].values.item()
        error_mae.append(error)
        error = compute_acc(persistence_fc, target)[var].values.item()
        error_acc.append(error) 
    df_error[var+'_RMSE'] = error_rmse
    df_error[var+'_MAE'] = error_mae
    df_error[var+'_ACC'] = error_acc
    


# In[73]:


df_error


# In[74]:


df_error.iloc[[2, 4]]


# In[83]:


df = pd.DataFrame({ 'Lead Days': np.arange(1, 16, 1),'RMSE': df_error['APCP_sfc_RMSE'], 'MAE': df_error['APCP_sfc_MAE'], 'ACC': df_error['APCP_sfc_ACC']})
df.head()


# In[84]:


fig, ax1 = plt.subplots()

plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold' )

ax1.plot(df['Lead Days'], df['RMSE'], label='RMSE', color='blue')
ax1.plot(df['Lead Days'], df['MAE'], label='MAE', color='green')
ax1.set_xlabel('Lead Days', fontweight='bold')
ax1.set_ylabel('ERROR (m)', color='black', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='black', labelsize='large', width=2 )
ax1.tick_params(axis='x', labelcolor='black', labelsize='large', width=2)

ax2 = ax1.twinx()
ax2.plot(df['Lead Days'], df['ACC'], label='ACC', color='red')
ax2.set_ylabel('ACC', color='red', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='red', labelsize='large', width=2)


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2

legend = ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.88))
# Make legend labels bold
for label in legend.get_texts():
    label.set_fontweight('bold')

 
ax1.set_title('6 hourly accumulate precipitation', fontweight='bold')

plt.yticks(fontweight='bold' )

plt.savefig(r'D:\VSCODE_Works\BharatBench\ignore\Figures\APCP_sfc_persistence.png', dpi=300)

plt.show()


# In[ ]:





# ## Climatology

# In[14]:


clim = ds.sel(time=train_years).groupby('time.dayofyear').mean()
error_rmse = compute_rmse(clim.sel(dayofyear=ds.sel(time=test_years).time.dt.dayofyear), ds.sel(time=test_years))
error_mae = compute_mae(clim.sel(dayofyear=ds.sel(time=test_years).time.dt.dayofyear), ds.sel(time=test_years))
error_acc = compute_acc(clim.sel(dayofyear=ds.sel(time=test_years).time.dt.dayofyear), ds.sel(time=test_years)) 


# In[15]:


df_error = pd.DataFrame()
for var in var_name:
    df_error[var] = pd.DataFrame({var: [error_rmse[var].values, error_mae[var].values, error_acc[var].values]}, index=['RMSE', 'MAE', 'ACC'])
df_error    


# In[16]:


print(f" Geopotential  Height at 500hPa \n RMSE : {error_rmse['HGT_prl'].values}\n MAE : {error_mae['HGT_prl'].values}\n ACC : {error_acc['HGT_prl'].values} ")
print(f" Temperature at  850hPa \n RMSE : {error_rmse['TMP_prl'].values}\n MAE : {error_mae['TMP_prl'].values}\n ACC : {error_acc['TMP_prl'].values} ")
print(f" Total Precipitation \n RMSE : {error_rmse['APCP_sfc'].values}\n MAE : {error_mae['APCP_sfc'].values}\n ACC : {error_acc['APCP_sfc'].values} ")
print(f" 2m Temperature \n RMSE : {error_rmse['TMP_2m'].values}\n MAE : {error_mae['TMP_2m'].values}\n ACC : {error_acc['TMP_2m'].values} ")


# ## Weekly Climatology

# In[17]:


# computation of weekly climatology
def compute_weekly_climatology(ds_train, valid_time):
    ds_train['week'] = ds_train['time.week']
    weekly_averages = ds_train.groupby('week').mean('time')
    valid_time['week'] = valid_time['time.week']
    fc_list = []
    for t in valid_time:
        fc_list.append(weekly_averages.sel(week=t.week))
    return xr.concat(fc_list, dim=valid_time)


# In[18]:


train_data = ds.sel(time=train_years)
target = ds.sel(time=test_years)


# In[19]:


weekly_climatology = compute_weekly_climatology(train_data, target.time)


# In[20]:


error_rmse = compute_rmse(weekly_climatology, target)
error_mae = compute_mae(weekly_climatology, target)
error_acc = compute_acc(weekly_climatology, target)


# In[21]:


df_error = pd.DataFrame()
for var in var_name:
    df_error[var] = pd.DataFrame({var: [error_rmse[var].values, error_mae[var].values, error_acc[var].values]}, index=['RMSE', 'MAE', 'ACC'])
df_error 


# In[22]:


print(f" Geopotential  Height at 500hPa \n RMSE : {error_rmse['HGT_prl'].values}\n MAE : {error_mae['HGT_prl'].values}\n ACC : {error_acc['HGT_prl'].values} ")
print(f" Temperature at  850hPa \n RMSE : {error_rmse['TMP_prl'].values}\n MAE : {error_mae['TMP_prl'].values}\n ACC : {error_acc['TMP_prl'].values} ")
print(f" Total Precipitation \n RMSE : {error_rmse['APCP_sfc'].values}\n MAE : {error_mae['APCP_sfc'].values}\n ACC : {error_acc['APCP_sfc'].values} ")
print(f" 2m Temperature \n RMSE : {error_rmse['TMP_2m'].values}\n MAE : {error_mae['TMP_2m'].values}\n ACC : {error_acc['TMP_2m'].values} ")

