!pip install visualkeras
!pip install xarray

# Libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

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

# Mount Google Drive
from google.colab import drive
drive.mount('/gdrive')

file_path = '/gdrive/My Drive/BharatBench/Datasets/IMDAA_merged_1.08_1990_2020.nc'
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

# Convert to PyTorch tensors for the Spherical CNN
X_train_torch = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)  # Channels first
Y_train_torch = torch.tensor(Y_train, dtype=torch.float32)
X_valid_torch = torch.tensor(X_valid, dtype=torch.float32).permute(0, 3, 1, 2)
Y_valid_torch = torch.tensor(Y_valid, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
Y_test_torch = torch.tensor(Y_test, dtype=torch.float32)

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
    keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
    # keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
    keras.layers.MaxPooling2D(),

    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation= 'selu'),
    keras.layers.UpSampling2D(),
    # keras.layers.BatchNormalization(),

    keras.layers.Conv2D(1, 5, padding='same'),

    # No activation since we are solving a regression problem
])

model_cnn.build(X_train[:32].shape)
model_cnn.compile(keras.optimizers.Adam(learning_rate=1e-5), 'mse')
model_cnn.summary()

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, verbose=1)
history_cnn = fit_model(model_cnn)


variables = ['HGT_prl', 'TMP_prl', 'TMP_2m', 'APCP_sfc']

class BasicCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256 * 8 * 8, out_channels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Initialize the model with in_channels matching the number of input channels
model_scnn = SphericalCNN(in_channels=1, out_channels=32)

# Initialize the model
model_cnn = BasicCNN(in_channels=1, out_channels=1)


def fit_spherical_model(model, X_train, Y_train, X_valid, Y_valid, epochs=10, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_output = model(X_valid)
            valid_loss = criterion(valid_output, Y_valid)
        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {valid_loss.item()}')


# Initialize and train the Spherical CNN model

print("X_train_torch shape:", X_train_torch.shape)
print("Y_train_torch shape:", Y_train_torch.shape)
print("X_valid_torch shape:", X_valid_torch.shape)
print("Y_valid_torch shape:", Y_valid_torch.shape)




model_scnn = SphericalCNN(in_channels=len(variables), out_channels=32)
print("Model FieldType input size:", model_scnn.input_type.size)

fit_spherical_model(model_scnn, X_train_torch, Y_train_torch, X_valid_torch, Y_valid_torch)

# Evaluation
model_scnn.eval()
with torch.no_grad():
    pred_test_scnn = model_scnn(X_test_torch).numpy()

# Convert predictions back to original scale if needed
pred_test_scnn_rescaled = pred_test_scnn * std.values + mean.values

# Compute RMSE, MAE, ACC for the Spherical CNN
rmse_scnn = compute_rmse(pred_test_scnn_rescaled, Y_test)
mae_scnn = compute_mae(pred_test_scnn_rescaled, Y_test)
acc_scnn = compute_acc(pred_test_scnn_rescaled, Y_test)

print(f'Spherical CNN - RMSE: {rmse_scnn}, MAE: {mae_scnn}, ACC: {acc_scnn}')

# Performance of the standard CNN (already implemented)
target = ds.sel(time=test_years)
pred_test = X_test[:, :, :, 0].copy()
pred_test[:] = model_cnn.predict(X_test).squeeze()

# Unnormalize
pred_result = pred_test * std.TMP_prl.values + mean.TMP_prl.values
pred_result = xr.DataArray(pred_result, dims=target.isel(time=slice(lead_time_steps, None)).dims, coords=target.isel(time=slice(lead_time_steps, None)).coords)

# Compute RMSE, MAE, ACC for the standard CNN
rmse_cnn = compute_rmse(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values
mae_cnn = compute_mae(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values
acc_cnn = compute_acc(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values

print(f'CNN - RMSE: {rmse_cnn}, MAE: {mae_cnn}, ACC: {acc_cnn}')

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

# # plot training history
# # print("Values stored in history are ... \n", history_cnn.history)
# plt.plot(history_cnn.history['loss'], label='train')
# plt.plot(history_cnn.history['val_loss'], label='test')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # # Visualize the architecture of the CNN
# # tf.keras.utils.plot_model(model, to_file= "CNN_IMDAA.png", show_shapes=True)
# # Visualize the architecture of the CNN
# keras.utils.plot_model(model_cnn, show_shapes=True)

# visualkeras.layered_view(model_cnn, legend=True, draw_volume = 2)

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
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'swish'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'swish'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'swish'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'swish'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'swish'),
    keras.layers.UpSampling3D(size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'swish'),
    keras.layers.UpSampling3D(size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'swish'),
    keras.layers.UpSampling3D(size=(1, 2, 2)),
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=True, activation= 'swish'),
    keras.layers.UpSampling3D(size=(1, 2, 2)),

    keras.layers.Conv3D(filters=1, kernel_size=(5, 5, 5),  padding="same")
])

model_conv_lstm.build((None, 1, 32, 32,  1))
model_conv_lstm.compile(keras.optimizers.Adam(learning_rate=1e-6), 'mse')
model_conv_lstm.summary()

# #create callback
# filepath = 'D:/VSCODE_Works/BharatBench/ignore/IMDAA_convlstm_H500_3days.hdf5'
# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_loss',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='min')

# early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 3, verbose=1)

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, verbose=1)
def fit_model(model):

    history = model.fit(X_train, Y_train, epochs = 1,
                        validation_data= (X_valid, Y_valid) ,
                        batch_size = 32, shuffle = False,
                        callbacks = [early_stop])
    return history
history_conv_lstm = fit_model(model_conv_lstm)

# model.save('IMDAA_convlstm_T850_5days_val_loss_0.1728.hdf5')

# model = load_model('IMDAA_convlstm_T850_5days_val_loss_0.1728.hdf5')

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


