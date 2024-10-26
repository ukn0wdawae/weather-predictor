!pip install visualkeras
!pip install xarray

# Libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import visualkeras
import tensorflow
from tensorflow import keras as keras
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model

from keras import Model, Sequential

from keras.optimizers import Adam

from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

from keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, GRU, LSTMCell
from keras.layers import Conv2D, MaxPooling2D, Flatten, Bidirectional, LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Dropout, Conv3D
import warnings
warnings.filterwarnings('ignore')

# Alternate approach could be to mount, this one uses a google drive link to download dataset
!pip install gdown
import gdown

# Download the file from a google drive link
file_id = '1wqsx6FcNKAVjBaMdRHN5ccFOZgenUbap'
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'IMDAA_merged_1.08_1990_2020.nc', quiet=False)

file_path = 'IMDAA_merged_1.08_1990_2020.nc'
data=  xr.open_dataset(file_path)
data.head()

data.attrs

for var in data.data_vars:
    data[var].isel(time=0).plot()
    plt.title(var)
    plt.show()
    print(var)

var_name = ['HGT_prl', 'TMP_prl', 'TMP_2m', 'APCP_sfc'] # [H500, T850, T2m, TP6h]

ds = data['TMP_prl']
ds = ds.to_dataset()
ds

# training dataset selection
train_years = slice('1990', '2017')
# validation dataset selection (this dataset helps with overfitting)
valid_years = slice('2018', '2018')
# test dataset selection
test_years = slice('2019', '2020')
# prediction days ahead
lead_time_steps = 20 # consider the number of observations per day

def get_train_valid_test_dataset(lead_steps, Data_array):
  # Split train, valid and test dataset
  train_data = Data_array.sel(time=train_years)
  valid_data = Data_array.sel(time=valid_years)
  test_data = Data_array.sel(time=test_years)

  # Normalize the data using the mean and standard deviation of the training data
  # mean = train_data.mean(dim = "time")
  # std = train_data.std(dim = "time")

  mean = train_data.mean()
  std = train_data.std()

  train_data = (train_data - mean) / std
  valid_data = (valid_data - mean) / std
  test_data = (test_data - mean) / std

  # Create inputs and outputs that are shifted by lead_steps
  X_train = train_data[list(Data_array)[0]].isel(time=slice(None, -lead_steps)).values[..., None]
  Y_train = train_data[list(Data_array)[0]].isel(time=slice(lead_steps, None)).values[..., None]
  X_valid = valid_data[list(Data_array)[0]].isel(time=slice(None, -lead_steps)).values[..., None]
  Y_valid = valid_data[list(Data_array)[0]].isel(time=slice(lead_steps, None)).values[..., None]
  X_test = test_data[list(Data_array)[0]].isel(time=slice(None, -lead_steps)).values[..., None]
  Y_test = test_data[list(Data_array)[0]].isel(time=slice(lead_steps, None)).values[..., None]
  return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, mean, std

X_train, Y_train, X_valid, Y_valid, X_test, Y_test, mean, std = get_train_valid_test_dataset(lead_time_steps, ds)

mean

std

print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)
print(X_test.shape)
print(Y_test.shape)

def compute_rmse(prediction, actual,  mean_dims = ('time', 'latitude', 'longitude')):
  error = prediction - actual
  rmse = np.sqrt(((error)**2 ).mean(mean_dims))
  return rmse

def compute_mae(prediction, actual, mean_dims = ('time', 'latitude', 'longitude')):
    error = prediction - actual
    mae = np.abs(error).mean(mean_dims)
    return mae

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

# Check the size of training, validation and testing dataset
print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)
print(X_test.shape)
print(Y_test.shape)

def fit_model(model):

    history = model.fit(X_train, Y_train, epochs = 10,
                        validation_data= (X_valid, Y_valid) ,
                        batch_size = 32, shuffle = False,
                        callbacks = [early_stop])
    return history

model_cnn = keras.Sequential([
    keras.layers.Conv2D(32, 5, padding='same', activation= 'swish'),
    # keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'swish'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'swish'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'swish'),
    keras.layers.MaxPooling2D(),

    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(32, 5, padding='same', activation= 'swish'),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'swish'),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'swish'),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'swish'),
    keras.layers.UpSampling2D(),
    # keras.layers.BatchNormalization(),

    keras.layers.Conv2D(1, 5, padding='same'),

    # No activation since we are solving a regression problem
])

model_cnn.build(X_train[:32].shape)
model_cnn.compile(keras.optimizers.Adam(learning_rate=1e-5), 'mse')
model_cnn.summary()

#create callback
filepath = 'IMDAA_CNN_H500_5days.keras'

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, verbose=1)
history_cnn = fit_model(model_cnn)

# save the model
model_cnn.save('IMDAA_CNN_T850_5days_val_loss_0.1473.keras')

# plot training history
print("Values stored in history are ... \n", history_cnn.history)
plt.plot(history_cnn.history['loss'], label='train')
plt.plot(history_cnn.history['val_loss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize the architecture of the CNN
# tf.keras.utils.plot_model(model, to_file= "CNN_IMDAA.png", show_shapes=True)
# Visualize the architecture of the CNN
keras.utils.plot_model(model_cnn, show_shapes=True)

visualkeras.layered_view(model_cnn, legend=True, draw_volume = 2)

target = ds.sel(time=test_years)
# Convert predictions backto xarray
pred_test = X_test[:, :, :, 0].copy()
pred_test[:] = model_cnn.predict(X_test).squeeze()

# Unnormalize
pred_result = pred_test*std.TMP_prl.values + mean.TMP_prl.values
pred_result  = xr.DataArray(pred_result, dims=target.isel(time=slice(lead_time_steps, None)).dims, coords=target.isel(time=slice(lead_time_steps, None)).coords)
# compute RMSE
print('RMSE:', compute_rmse(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)
print('MAE', compute_mae(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)
print('ACC', compute_acc(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)

def build_model(activation):
  model_cnn_new = keras.Sequential([
      keras.layers.Conv2D(32, 5, padding='same', activation= activation),
      keras.layers.MaxPooling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= activation),
      keras.layers.MaxPooling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= activation),
      keras.layers.MaxPooling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= activation),
      keras.layers.MaxPooling2D(),

      keras.layers.Dropout(0.2),

      keras.layers.Conv2D(32, 5, padding='same', activation= activation),
      keras.layers.UpSampling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= activation),
      keras.layers.UpSampling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= activation),
      keras.layers.UpSampling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= activation),
      keras.layers.UpSampling2D(),

      keras.layers.Conv2D(1, 5, padding='same'),

      # No activation since we are solving a regression problem
  ])
  return model_cnn_new

def test_model(activation):
  model_cnn_new = build_model(activation)
  model_cnn_new.build(X_train[:32].shape)
  model_cnn_new.compile(keras.optimizers.Adam(learning_rate=1e-5), 'mse')

  history_cnn_new = fit_model(model_cnn_new)


  # Evaluation

  target = ds.sel(time=test_years)
  # Convert predictions backto xarray
  pred_test = X_test[:, :, :, 0].copy()
  pred_test[:] = model_cnn_new.predict(X_test).squeeze()

  # Unnormalize
  pred_result_new = pred_test*std.TMP_prl.values + mean.TMP_prl.values
  pred_result_new  = xr.DataArray(pred_result_new, dims=target.isel(time=slice(lead_time_steps, None)).dims, coords=target.isel(time=slice(lead_time_steps, None)).coords)
  # compute RMSE
  return pred_result_new
  # model_cnn_new.summary()

# Comparison of CNN model's behaviour with different activation types
keras_conv2d_activations = [
    'relu',
    'sigmoid',
    'softmax',
    'softplus',
    'softsign',
    'tanh',
    'selu',
    'elu',
    'exponential',
    'leaky_relu',
    'relu6',
    'silu',
    'hard_silu',
    'gelu',
    'hard_sigmoid',
    'linear',
    'mish',
    'log_softmax',
    'swish',
    'hard_swish'
]
# Stores model performance for our final comparison
lowest_rmse = (100,"")
lowest_mae = (100,"")
highest_acc = (0,"")
target = ds.sel(time=test_years)
early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, verbose=1)
for activation in keras_conv2d_activations:
  print("##########     "+activation+"        ##############")
  pred_result_new = test_model(activation)
  print("##########     "+activation+"        ##############")
  rmse = compute_rmse(pred_result_new, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values
  mae = compute_mae(pred_result_new, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values
  acc = compute_acc(pred_result_new, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values
  print("RMSE:", rmse)
  print("MAE", mae)
  print("ACC", acc)
  if(rmse < lowest_rmse[0]):
    lowest_rmse = (rmse,activation)
  if(mae < lowest_mae[0]):
    lowest_mae = (mae,activation)
  if(acc > highest_acc[0]):
    highest_acc = (acc,activation)

print("Best models for each metric:")
print(f"RMSE: {lowest_rmse}")
print(f"MAE: {lowest_mae}")
print(f"ACC: {highest_acc}")

# Choosing activation selu due to performance
model_cnn_new = keras.Sequential([
      # Layer 1: filter size = 10, number of filters = 128
      keras.layers.Conv2D(512, 10, padding='same', activation= 'selu'),
      keras.layers.MaxPooling2D(),
      # Layer 2: filter size = 5, number of filters = 32
      keras.layers.Conv2D(64, 5, padding='same', activation= 'selu'),
      keras.layers.MaxPooling2D(),
      # Layer 3: filter size = 5, number of filters = 64
      keras.layers.Conv2D(128, 5, padding='same', activation= 'selu'),
      keras.layers.MaxPooling2D(),
      # # Layer 4: filter size = 10, number of filters = 32
      # keras.layers.Conv2D(32, 10, padding='same', activation= 'selu'),
      # keras.layers.MaxPooling2D(),

      keras.layers.Dropout(0.3),

      # keras.layers.Conv2D(32, 10, padding='same', activation= 'selu'),
      # keras.layers.UpSampling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
      keras.layers.UpSampling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
      keras.layers.UpSampling2D(),
      keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
      keras.layers.UpSampling2D(),

      keras.layers.Conv2D(1, 5, padding='same'),

      # No activation since we are solving a regression problem
  ])

model_cnn_new.build(X_train[:32].shape)
model_cnn_new.compile(keras.optimizers.Adam(learning_rate=1e-4), 'mse')
model_cnn_new.summary()

#create callback
filepath = 'IMDAA_CNN_H500_5days_new.keras'

checkpoint_new = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, verbose=1)
history_cnn_new = fit_model(model_cnn_new)

# save the model
model_cnn_new.save('IMDAA_CNN_T850_5days_val_loss_0.1473_new.keras')

# plot training history
print("Values stored in history are ... \n", history_cnn_new.history)
plt.plot(history_cnn_new.history['loss'], label='train')
plt.plot(history_cnn_new.history['val_loss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize the architecture of the CNN
keras.utils.plot_model(model_cnn_new, show_shapes=True)

visualkeras.layered_view(model_cnn_new, legend=True, draw_volume = 2)

target = ds.sel(time=test_years)
# Convert predictions backto xarray
pred_test = X_test[:, :, :, 0].copy()
pred_test[:] = model_cnn_new.predict(X_test).squeeze()

# Unnormalize
pred_result = pred_test*std.TMP_prl.values + mean.TMP_prl.values
pred_result  = xr.DataArray(pred_result, dims=target.isel(time=slice(lead_time_steps, None)).dims, coords=target.isel(time=slice(lead_time_steps, None)).coords)
# compute RMSE
print('RMSE:', compute_rmse(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)
print('MAE', compute_mae(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)
print('ACC', compute_acc(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)

X_train = X_train[:, np.newaxis,:,:,:]
Y_train = Y_train[:, np.newaxis,:,:,:]
X_valid = X_valid[:, np.newaxis,:,:,:]
Y_valid = Y_valid[:, np.newaxis,:,:,:]
X_test = X_test[:, np.newaxis,:,:,:]
Y_test = Y_test[:, np.newaxis,:,:,:]

print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)
print(X_test.shape)
print(Y_test.shape)

# Build an convLSTM network
model_conv_lstm = keras.Sequential([
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'selu'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'selu'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'selu'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'selu'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'selu'),
    keras.layers.UpSampling3D(size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'selu'),
    keras.layers.UpSampling3D(size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'selu'),
    keras.layers.UpSampling3D(size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'selu'),
    keras.layers.UpSampling3D(size=(1, 2, 2)),

    keras.layers.Conv3D(filters=1, kernel_size=(5, 5, 5),  padding="same")
])

model_conv_lstm.build((None, 1, 32, 32,  1))
model_conv_lstm.compile(keras.optimizers.Adam(learning_rate=1e-6), 'mse')
model_conv_lstm.summary()

#create callback
filepath = 'IMDAA_convlstm_H500_3days.keras'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 3, verbose=1)

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, verbose=1)
def fit_model(model):

    history = model.fit(X_train, Y_train, epochs = 1,
                        validation_data= (X_valid, Y_valid) ,
                        batch_size = 32, shuffle = False,
                        callbacks = [early_stop])
    return history
history_conv_lstm = fit_model(model_conv_lstm)

model_conv_lstm.save('IMDAA_convlstm_T850_5days_val_loss_0.1728.keras')

model_conv_lstm = load_model('IMDAA_convlstm_T850_5days_val_loss_0.1728.keras')

target = ds.sel(time=test_years)

pred_test = model_conv_lstm.predict(X_test).squeeze()

# Unnormalize
pred_result = pred_test*std.TMP_prl.values + mean.TMP_prl.values
pred_result  = xr.DataArray(pred_result, dims=target.isel(time=slice(lead_time_steps, None)).dims, coords=target.isel(time=slice(lead_time_steps, None)).coords)
# compute RMSE
print('RMSE:', compute_rmse(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)
print('MAE', compute_mae(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)
print('ACC', compute_acc(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values)

# keras.utils.plot_model(model_conv_lstm, show_shapes=True)

visualkeras.layered_view(model_conv_lstm, legend=True, draw_volume = 1)


