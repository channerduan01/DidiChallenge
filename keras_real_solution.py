import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import linear_model  

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, Adamax
from keras.regularizers import l2, l1, activity_l2, activity_l1
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.layers import Convolution1D, MaxPooling1D

from numpy import genfromtxt, savetxt
TRAIN_RATIO = 0.8

def shuffle_data(data_, label_):
    shuffle_data = np.zeros_like(data_, dtype=np.int)
    shuffle_label = np.zeros_like(label_, dtype=np.int)
    shuffle_idx = range(shuffle_data.shape[0])
    np.random.shuffle(shuffle_idx)
    for i in range(len(shuffle_idx)):
        idx_ = shuffle_idx[i]
        shuffle_data[i, :] = data_[idx_]
        shuffle_label[i, :] = label_[idx_]
    return (shuffle_data, shuffle_label)
    
TRAIN_DATA_FILEPATH = '/Users/channerduan/Desktop/didi_train_data.csv'
TRAIN_LABEL_FILEPATH = '/Users/channerduan/Desktop/didi_train_label.csv'
WEATHER_DATA_FILEPATH = '/Users/channerduan/Desktop/didi_weather.csv'
if not ('BASE_POI_NUM' in dir() and 'data_' in dir() and 'label_' in dir()):
    print 'load training data!'
    data_ = genfromtxt(open(TRAIN_DATA_FILEPATH,'r'), delimiter=',', dtype='int')
    label_ = genfromtxt(open(TRAIN_LABEL_FILEPATH,'r'), delimiter=',', dtype='int')
    weather_ = genfromtxt(open(WEATHER_DATA_FILEPATH,'r'), delimiter=',', dtype='int')
#    data_ = np.c_[data_,weather_[:,[0,3,6,9]]]
#    data_ = np.c_[data_,weather_[:,9]]
    BASE_POI_NUM = label_.shape[1]

(data_, label_) = shuffle_data(data_, label_)

data_list = []
for i in range(1,BASE_POI_NUM+1):
#    select_features = data_[:,[0,i,i+BASE_POI_NUM,i+2*BASE_POI_NUM]] # select partial features
    select_features = data_
    select_label_idx = np.where(label_[:,i-1]>0)[0] # select nonnegative datas
    data_i = select_features[select_label_idx,:]
    label_i = label_[select_label_idx,i-1]
    data_list.append((data_i,label_i))

class ProcessControl(Callback):  
    def on_train_begin(self, logs={}):  
        self.losses = []
    def on_epoch_begin(self, epoch, logs={}):
        new_lr = np.float32(model.optimizer.lr.get_value()*0.99)
        model.optimizer.lr.set_value(new_lr)
#        print new_lr
# training -------------------------------------------------
model_list = None        
if not 'model_list' in dir() or model_list == None:
    sum_all = 0
    number_all = 0
    model_list = []
    err_list = []
#    for i in range(len(data_list)):
    for i in range(0,1):
        sum_ = 0
        num_ = 0
        print 'calcu: %d' %i
        (my_data,my_label) = data_list[i]
#        my_data = my_data.reshape((my_data.shape[0],my_data.shape[1],1))
        
        train_num = int(np.floor(len(my_data)*TRAIN_RATIO))
        train_data = my_data[0:train_num]
        test_data = my_data[train_num:]
        train_label = my_label[0:train_num]
        test_label = my_label[train_num:]
        activator = 'linear'
#        activator = 'sigmoid'
        model = Sequential()
        weight_dacay = 0.01;
        node_num = 20
        
#        model.add(Convolution1D(32, 6, border_mode='valid', activation=activator,
#                                input_shape=(199,1)))
#        model.add(Dropout(0.5))
#        model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
#        model.add(Flatten())
        model.add(Dense(node_num,activation=activator, input_dim=train_data.shape[1]
            ,W_regularizer=l2(weight_dacay)))
#            ,activity_regularizer=activity_l2(0.0001)))
        model.add(Dropout(0.5))
        model.add(Dense(node_num, activation=activator
            ,W_regularizer=l2(weight_dacay)))
#            ,activity_regularizer=activity_l2(0.0001)))
        model.add(Dropout(0.5))
        
        model.add(Dense(1))
        adam = Adam(lr=0.001, beta_1=0.9,beta_2=0.999,epsilon=1e-08)
        model.compile(loss='mape', optimizer=adam)
        earlystopping = EarlyStopping(monitor='loss', patience=12, verbose=False, mode='min')
        model.fit(train_data, train_label, shuffle=True, validation_split=0.1, 
                  verbose=False, nb_epoch=100, batch_size=16, callbacks=[ProcessControl(), earlystopping])
#        model.fit(train_data, train_label, shuffle=True,
#                  verbose=False, nb_epoch=80, batch_size=16, callbacks=[earlystopping])
        
        model_list.append(model)
        predict_result = model.predict(test_data).reshape(len(test_label))
        a = np.abs(test_label-predict_result)
        for j in range(len(predict_result)):
            if test_label[j] != 0:
                sum_ = sum_ + a[j]/test_label[j]
                num_ = num_ + 1
        print '%d-mape: %f\n' %(i+1,sum_/num_)
        err_list.append(sum_/num_)
        sum_all = sum_all+sum_
        number_all = number_all+num_
    print 'total mape: %f\n' %(sum_all/number_all)

## predict -------------------------------------------------
#OUTPUT_TEST_FEATURE_PATH_FILEPATH = '/Users/channerduan/Desktop/didi_test_data.csv'
#OUTPUT_RESULT_FILEPATH = '/Users/channerduan/Desktop/didi.csv'
#time_str = '2016-01-22-46,2016-01-22-58,2016-01-22-70,2016-01-22-82,2016-01-22-94,2016-01-22-106,2016-01-22-118,2016-01-22-130,2016-01-22-142,2016-01-24-58,2016-01-24-70,2016-01-24-82,2016-01-24-94,2016-01-24-106,2016-01-24-118,2016-01-24-130,2016-01-24-142,2016-01-26-46,2016-01-26-58,2016-01-26-70,2016-01-26-82,2016-01-26-94,2016-01-26-106,2016-01-26-118,2016-01-26-130,2016-01-26-142,2016-01-28-58,2016-01-28-70,2016-01-28-82,2016-01-28-94,2016-01-28-106,2016-01-28-118,2016-01-28-130,2016-01-28-142,2016-01-30-46,2016-01-30-58,2016-01-30-70,2016-01-30-82,2016-01-30-94,2016-01-30-106,2016-01-30-118,2016-01-30-130,2016-01-30-142'
#list_slot_str = time_str.split(',')
#test_data = genfromtxt(open(OUTPUT_TEST_FEATURE_PATH_FILEPATH,'r'), delimiter=',', dtype='int')
#slot_num = test_data.shape[0]
#f_out = open(OUTPUT_RESULT_FILEPATH,'w')
#res_list = []
#for i in range(1,BASE_POI_NUM+1):
#    select_ = test_data[:,[0,i,i+BASE_POI_NUM,i+2*BASE_POI_NUM]]
#    res_list.append(model_list[i-1].predict(select_))
#for i in range(slot_num):
#    for j in range(BASE_POI_NUM):
#        f_out.write('%d,%s,%f\n' %(j+1,list_slot_str[i],res_list[j][i,0]))
#f_out.close()






