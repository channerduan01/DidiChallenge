import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import linear_model  

import keras
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Flatten
from keras.regularizers import l2, activity_l2
from keras.layers.recurrent import LSTM
  
population = [54167,55196,56300,57482,58796,60266,61465,62828,64653,65994,67207,66207,65859,67295,69172,70499,72538,74542,76368,78534,80671,82992,85229,87177,89211,90859,92420,93717,94974,96259,97542,98705,100072,101654,103008,104357,105851,107507,109300,111026,112704,114333,115823,117171,118517,119850,121121,122389,123626,124761,125786,126743,127627,128453,129227,129988,130756,131448,132129,132802,134480,135030,135770,136460,137510]  
lag = 3
N = len(population)-lag
train_data = np.zeros((N,lag), dtype=np.int)
train_label = np.zeros((N,1), dtype=np.int)
for i in range(N):
    train_data[i,:] = population[i:i+lag]
    train_label[i] = population[i+lag]


# Linear Regression -------------------------------------------------------
#clf = linear_model.LinearRegression()   
#clf.fit(train_data, train_label)
#predict_ = clf.predict(train_data)
#print 'eva: %f' %(np.sum(np.abs(predict_-train_label))/len(train_label))

# Simple ANN --------------------------------------------------------------
#model = Sequential()
#model.add(Dense(20, input_dim=lag,activation='relu'))
#model.add(Dense(20, activation='relu'))
#model.add(Dense(1))
#model.compile(loss='mse', optimizer='adam')
#model.fit(train_data, train_label, nb_epoch=100, batch_size=16)
#predict_ = model.predict(train_data)
#print 'eva: %f' %(np.sum(np.abs(predict_-train_label))/len(train_label))

# RNN LSTM ----------------------------------------------------------------
activator = 'linear'
timesteps = lag
model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, 1), return_sequences=True, W_regularizer=l2(0.05)))
model.add(Activation(activator))
model.add(Dropout(0.5))
model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))

model.add(LSTM(64, return_sequences=True, W_regularizer=l2(0.05)))
model.add(Activation(activator))
model.add(Dropout(0.5))
model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))
    
model.add(Flatten())

model.add(Dense(64, W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)))
model.add(Dropout(0.5))
model.add(Activation(activator))

model.add(Dense(32, W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)))
model.add(Dropout(0.5))
model.add(Activation(activator))

model.add(Dense(16, W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)))
model.add(Dropout(0.5))
model.add(Activation(activator))

model.add(Dense(1, W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)))
model.add(Activation(activator))    

optimizer = keras.optimizers.Adam(lr=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mape', optimizer=optimizer)

train_data_lstm = np.zeros((N,timesteps,1),dtype=np.int)
for i in range(train_data_lstm.shape[0]):
    train_data_lstm[i,:,:] = train_data[0].reshape((3,1))
model.fit(train_data_lstm, train_label, verbose=True, batch_size=16, nb_epoch=40)
predicted = model.predict(train_data)


