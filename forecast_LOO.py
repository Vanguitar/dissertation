from keras.layers import Dropout, Dense, GRU,LSTM,SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import scipy.io as sio
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import LeaveOneOut
from math import sqrt
from tkinter import _flatten
import h5py
from keras.utils import plot_model
import multiprocessing as mp
''' 1/3/5step  '''
def loadmat1(dfile):
    spe = sio.loadmat(dfile)
    data = spe['dchi']
    data11 = data.flatten()
    data = data11.tolist()
    data1 = {'one': data}
    yt = pd.DataFrame(data1)
    print(yt.shape)
    yt_1 = yt.shift(1)
    data2 = pd.concat([yt, yt_1], axis=1)
    data2.columns = ['yt', 'yt_1']
    data3 = data2.dropna()  # delete NULL，
    # print('2:',data3)
    x = data3[['yt_1']]
    y = data3['yt']
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    x = np.array(x).reshape((len(x), 1))
    x = scaler_x.fit_transform(x)  # (265, 3)
    scaler_y = MinMaxScaler(
        feature_range=(-1, 1))
    y = np.array(y).reshape((len(y), 1))
    y = scaler_y.fit_transform(y)
    return x,y, scaler_y

def loadmat3(dfile):
    spe = sio.loadmat(dfile)
    data = spe['dchi']
    data11 = data.flatten()
    data = data11.tolist()
    data1 = {'one': data}
    yt = pd.DataFrame(data1)
    print(yt.shape)
    yt_1 = yt.shift(1)
    yt_2 = yt.shift(2)
    yt_3 = yt.shift(3)
    data2 = pd.concat([yt, yt_1, yt_2, yt_3], axis=1)
    data2.columns = ['yt', 'yt_1', 'yt_2', 'yt_3']
    data3 = data2.dropna()  # delete NULL，
    # print('2:',data3)
    x = data3[['yt_1', 'yt_2', 'yt_3']]
    y = data3['yt']
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    x = np.array(x).reshape((len(x), 3))
    x = scaler_x.fit_transform(x)  # (265, 3)
    scaler_y = MinMaxScaler(
        feature_range=(-1, 1))
    y = np.array(y).reshape((len(y), 1))
    y = scaler_y.fit_transform(y)
    return x,y, scaler_y

def loadmat5(dfile):

    # spe = h5py.File(dfile)
    # data=spe['dchi'][:]
    spe = sio.loadmat(dfile)
    data = spe['dchi']
    data11 = data.flatten()
    data = data11.tolist()
    data1 = {'one': data}
    yt = pd.DataFrame(data1)
    print(yt.shape)
    yt_1 = yt.shift(1)
    yt_2 = yt.shift(2)
    yt_3 = yt.shift(3)
    yt_4 = yt.shift(4)
    yt_5 = yt.shift(5)
    data2 = pd.concat([yt, yt_1, yt_2, yt_3, yt_4, yt_5], axis=1)
    data2.columns = ['yt', 'yt_1', 'yt_2', 'yt_3', 'yt_4', 'yt_5']
    data3 = data2.dropna()  # delete NULL，
    # print('2:',data3)
    x = data3[['yt_1', 'yt_2', 'yt_3', 'yt_4', 'yt_5']]
    y = data3['yt']
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    x = np.array(x).reshape((len(x), 5))
    x = scaler_x.fit_transform(x)  # (265, 3)
    scaler_y = MinMaxScaler(
        feature_range=(-1, 1))
    y = np.array(y).reshape((len(y), 1))
    y = scaler_y.fit_transform(y)
    return x,y, scaler_y

def GRUmodel(x_train):

    model = Sequential()
    model.add(GRU(units=100, return_sequences=True, input_dim=x_train.shape[2], input_length=x_train.shape[1]))
    model.add(GRU(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def leaveone(x,y,epochs,batch_size):
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train = x_train . reshape ( x_train.shape[0],x_train.shape[1],1)  # (214, 3, 1)
        x_test = x_test . reshape ( x_test.shape[0],x_test.shape[1],1)
        #  build model
        model1=GRUmodel(x_train)
        model1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        preds=model1.predict(x_test)
        preds = scaler_y.inverse_transform(np.array(preds))
        y_test = scaler_y.inverse_transform(np.array(y_test))
        indexx=test_index.reshape(1,1) ##
        he = np.concatenate((indexx,y_test,preds))
        sio.savemat(str(test_index)+'.mat', {'GRUf25leave': he})



if __name__ == '__main__':

  dfile = 'F:/featureF1.mat'
  x,y, scaler_y=loadmat5(dfile)
  epochs = 1
  batch_size = 1

  leaveone(x, y, epochs, batch_size)

print("finished")
