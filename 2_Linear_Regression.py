#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# # Evaluation Metrics

# In[2]:


def compute_rmse(prediction, actual,  mean_dims = ('time', 'latitude', 'longitude')):
  error = prediction - actual
  rmse = np.sqrt(((error)**2 ).mean(mean_dims))
  return rmse


# In[3]:


def compute_mae(prediction, actual, mean_dims = ('time', 'latitude', 'longitude')):
    error = prediction - actual
    mae = np.abs(error).mean(mean_dims)
    return mae


# In[4]:


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


# # Dataset

# In[5]:


data =  xr.open_dataset(r"G:/IMDAA_Regrid_1.08_1990_2022/IMDAA_merged_1.08_1990_2020.nc")
data


# In[6]:


data_train = data.sel(time=slice('1990', '2018'))
data_test = data.sel(time=slice('2019', '2020'))


# In[7]:


test_data = data.sel(time=slice('2019', '2020'))


# In[8]:


data_mean = data_train.mean().load()
data_std = data_train.std().load()


# In[9]:


data_std


# In[10]:


# Normalize datasets
data_train = (data_train - data_mean) / data_std
data_test = (data_test - data_mean) / data_std


# In[11]:


_, nlat, nlon = data_train.HGT_prl.shape; nlat, nlon


# In[13]:


data_train


# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[15]:


def create_training_data(da, lead_time_h, return_valid_time=False):
    """Function to split input and output by lead time."""
    X = da.isel(time=slice(0, -lead_time_h))
    y = da.isel(time=slice(lead_time_h, None))
    valid_time = y.time
    if return_valid_time:
        return X.values.reshape(-1, nlat*nlon), y.values.reshape(-1, nlat*nlon), valid_time
    else:
        return X.values.reshape(-1, nlat*nlon), y.values.reshape(-1, nlat*nlon)


# In[16]:


def train_lr(lead_time_h, input_vars, output_vars, data_subsample=1):
    """Create data, train a linear regression and return the predictions."""
    X_train, y_train, X_test, y_test = [], [], [], []
    for v in input_vars:
        X, y = create_training_data(
            data_train[v],
            lead_time_h
        )

        X_train.append(X)
        if v in output_vars: y_train.append(y)
        X, y, valid_time = create_training_data(data_test[v], lead_time_h, return_valid_time=True)
        X_test.append(X)
        if v in output_vars: y_test.append(y)
    X_train, y_train, X_test, y_test = [np.concatenate(d, 1) for d in [X_train, y_train, X_test, y_test]]
    

    X_train = X_train[::data_subsample]
    y_train = y_train[::data_subsample]
    
 
    lr = LinearRegression(n_jobs=16)
    lr.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, lr.predict(X_train))
    mse_test = mean_squared_error(y_test, lr.predict(X_test))
    print(f'Train MSE = {mse_train}'); print(f'Test MSE = {mse_test}')
    preds = lr.predict(X_test).reshape((-1, len(output_vars), nlat, nlon))
  

    fcs = []
    for i, v in enumerate(output_vars):
        fc = xr.DataArray(
            preds[:, i] * data_std[v].values + data_mean[v].values,
            dims=['time', 'latitude', 'longitude'],
            coords={
                'time': valid_time,
                'lat': data_train.lat,
                'lon': data_train.lon
            },
            name=v
        )
        fcs.append(fc)
    return xr.merge(fcs), lr


# In[17]:


var_name = ['HGT_prl', 'TMP_prl', 'TMP_2m', 'APCP_sfc'] # [H500, T850, T2m, TP6h]


# In[18]:


experiments = [
    [['HGT_prl'], ['HGT_prl']],
    [['TMP_prl'], ['TMP_prl']],
    # [['HGT_prl', 'TMP_prl'], ['HGT_prl', 'TMP_prl']],
    [['APCP_sfc'], ['APCP_sfc']],
    # [['HGT_prl', 'TMP_prl', 'APCP_sfc'], ['APCP_sfc']],
    [['TMP_2m'], ['TMP_2m']],
    # [['HGT_prl', 'TMP_prl', 'TMP_2m'], ['TMP_2m']],
]


# In[29]:


data_subsample = 1
lead_time = 3*4
preds = []
models = []
df_error = pd.DataFrame()
for n, (i, o) in enumerate(experiments):
    # print(f'{n}: Input variables = {i}; output variables = {o}')
    var_name = o[0]
    p, m = train_lr(lead_time, input_vars=i, output_vars=o, data_subsample=data_subsample)
    preds.append(p); models.append(m)
    r = compute_rmse(p, test_data).compute()
    m = compute_mae(p, test_data).compute()
    a = compute_acc(p, test_data).compute()
    df_error[var_name ] = pd.DataFrame({var_name : [r[var_name].values, m[var_name].values, a[var_name].values]}, index=['RMSE', 'MAE', 'ACC'])
    
    #print('; '.join([f'{v} = {r[v].values}' for v in r]) + '\n')


# In[30]:


df_error

