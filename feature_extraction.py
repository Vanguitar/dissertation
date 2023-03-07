from numpy import array
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from keras.utils import plot_model
import h5py
import pandas as pd
import scipy.io as sio
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt

dfile = 'F:/F1.mat'
# data1 = h5py.File(dfile)
data1 = sio.loadmat(dfile)
data2 = data1['train_x']
data2 = data2.T
nn = data2.shape[1]
nn=int(nn/2)
trainx1 = data2[:,::2]
trainx1 = trainx1.transpose()
trainx2 = data2[:,1::2]
trainx2 = trainx2.transpose()

# normalization
scaler_x1 = MinMaxScaler (feature_range =(-1, 1))
trainx1 = np. array (trainx1)   #(417,15000)
trainx1 = scaler_x1 . fit_transform (trainx1)

scaler_x2 = MinMaxScaler (feature_range =(-1, 1))
trainx2 = np. array (trainx2)
trainx2 = scaler_x2 . fit_transform (trainx2)
# end

# threshold
trainx = np.dstack((trainx1,trainx2))
sequence = trainx[:5,:,:]
n_in = sequence.shape[1]

# model
model = Sequential()
model.add(GRU(200, activation='relu', input_shape=(n_in, 2)))
model.add(RepeatVector(n_in))
model.add(GRU(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(2)))

model.compile(optimizer='adam', loss='mse')
model.fit(sequence, sequence, epochs=1, verbose=1)
model.save('AGRN.h5')
plot_model(model, show_shapes=True, to_file='AGRN.png')
yhat = model.predict(sequence, verbose=0)

# format
ypred1 = np. array(yhat)[:,:,0].reshape(yhat.shape[0],yhat.shape[1])
ypred1 = scaler_x1.inverse_transform(ypred1)
ypred2 = np. array (yhat)[:,:,1].reshape(yhat.shape[0],yhat.shape[1])
ypred2 = scaler_x2 .inverse_transform (ypred2)
yhat111 = np.dstack((ypred1,ypred2))
# end

ee0 = yhat111 ** 2  # 1
ee1 = (sequence - yhat111) ** 2  # 2
ttlimi = (ee0 + ee1)**0.5  # +
ttlimi = np.sum(ttlimi,axis=1)  #
ttlimi = np.sum(ttlimi,axis=1)  #
ttlimi = np.array(ttlimi)
ttlim1 = ttlimi/1000
h = np.mean(ttlim1)
v = np.std(ttlim1, ddof=1)
g = v/(2*h)
alphaa = 0.95  # setting
dof=round(2*(h**2)/v)
k = chi2.ppf(alphaa, df=dof)
threshold1 = g*k
print(threshold1)
threshold2 = np.tile(threshold1,nn)

# all
data3 = trainx[:,:,:]  # (417, 15000, 2)
model = load_model('AGRN.h5')
plot_model(model, show_shapes=True, to_file='AGRN1.png')
yhat1 = model.predict(data3, verbose=1)

# format
ypred11 = np. array (yhat1)[:,:,0].reshape(yhat1.shape[0],yhat1.shape[1])
ypred11 = scaler_x1.inverse_transform(ypred11)
ypred22 = np. array (yhat1)[:,:,1].reshape(yhat1.shape[0],yhat1.shape[1])
ypred22 = scaler_x2.inverse_transform (ypred22)
yhat1111 = np.dstack((ypred11,ypred22))  # (417, 15000, 2)
# end

ee0 = yhat1111 ** 2  # 1
tt = np.sum(ee0,axis=1)  # (417, 2)
ee1 = (data3 - yhat1111) ** 2  # 2
chi = np.sum(ee1,axis=1)  # (417, 2)
ee2 = (ee0 + ee1)**0.5  # +
dchi = np.sum(ee2,axis=1)  # (417, 2)
dchi = np.sum(dchi,axis=1)  # (417,)
dchi = dchi/1000

sio.savemat('featureF1.mat', { 'dchi': dchi, 'threshold': threshold2})

