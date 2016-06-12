import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import linear_model  

import keras
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import Adam
from keras.regularizers import l2

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

from numpy import genfromtxt, savetxt

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
    
    
def evaluate(predict_result, test_label):
    sum_ = 0
    num_ = 0
    a = np.abs(test_label-predict_result)
    for j in range(len(predict_result)):
        if test_label[j] != 0:
            sum_ = sum_ + a[j]/test_label[j]
            num_ = num_ + 1
    return (sum_, num_)
    
TRAIN_DATA_FILEPATH = 'data/didi_train_data.csv'
TRAIN_LABEL_FILEPATH = 'data/didi_train_label.csv'
WEATHER_DATA_FILEPATH = 'data/didi_weather.csv'
if not ('BASE_POI_NUM' in dir() and 'data_' in dir() and 'label_' in dir() and data_ != None):
    print 'load training data!'
    data_ = genfromtxt(open(TRAIN_DATA_FILEPATH,'r'), delimiter=',', dtype='int')
    label_ = genfromtxt(open(TRAIN_LABEL_FILEPATH,'r'), delimiter=',', dtype='int')
    weather_ = genfromtxt(open(WEATHER_DATA_FILEPATH,'r'), delimiter=',', dtype='int')
    data_ = np.c_[data_,weather_[:,[0,3,6,9]]]
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

# training ----------------------------------------------------------------
class ProcessControl(Callback):  
    def on_train_begin(self, logs={}):  
        self.losses = []
    def on_epoch_begin(self, epoch, logs={}):
        new_lr = np.float32(model.optimizer.lr.get_value()*0.99)
        model.optimizer.lr.set_value(new_lr)         
TRAIN_RATIO = 0.7
VALID_RATIO = 0.9
#grid_base_config = [('linear',20), ('sigmoid',20), ('tanh',20), ('relu',20)]
grid_base_config = [('linear',20)]
grid_table = []
REPEAT_TIME = 1
for config in grid_base_config:
    for i in range(REPEAT_TIME):
        grid_table.append(config)

all_model_list = []
all_err_list = []
for (activator,layer_size) in grid_table:
    total_sum = 0
    total_num = 0
    model_list = []
    err_list = []
    for i in range(len(data_list)):
#    for i in range(0,1):
        sum_ = 0
        num_ = 0
        print 'calcu: %d' %i
        (my_data,my_label) = data_list[i]
        train_num = int(np.floor(len(my_data)*TRAIN_RATIO))
        valid_num = int(np.floor(len(my_data)*VALID_RATIO))
        train_data = my_data[0:train_num]
        train_label = my_label[0:train_num]
        valid_data = my_data[train_num:valid_num]
        valid_label = my_label[train_num:valid_num]        
        
        model = Sequential()
        weight_dacay = 0.01;
        model.add(Dense(layer_size, input_dim=train_data.shape[1],activation=activator
            ,W_regularizer=l2(weight_dacay)))
#        model.add(Dense(layer_size, input_dim=train_data.shape[1],activation=activator,W_regularizer=l1(0.1)))
        model.add(Dropout(0.5))
        model.add(Dense(layer_size, activation=activator
            ,W_regularizer=l2(weight_dacay)))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        adam = Adam(lr=0.001, beta_1=0.9,beta_2=0.999,epsilon=1e-08)
        model.compile(loss='mape', optimizer=adam)
        earlystopping = EarlyStopping(monitor='loss', patience=12, verbose=False, mode='min')
        model.fit(train_data, train_label, shuffle=True, validation_split=0.1, 
                  verbose=False, nb_epoch=100, batch_size=16, callbacks=[ProcessControl(), earlystopping])
        model_list.append(model)
        predict_result = model.predict(valid_data).reshape(len(valid_label))
        a = np.abs(valid_label-predict_result)
        for j in range(len(predict_result)):
            if valid_label[j] != 0:
                sum_ = sum_ + a[j]/valid_label[j]
                num_ = num_ + 1
        print '%d-mape: %f\n' %(i+1,sum_/num_)
        err_list.append((sum_,num_))
        total_sum = total_sum+sum_
        total_num = total_num+num_
    all_model_list.append(model_list)
    all_err_list.append(err_list)
    print 'total mape (%s - %d) : %f\n' %(activator, layer_size, total_sum/total_num)

#err_table = np.zeros((BASE_POI_NUM, len(grid_base_config)))
#for i in range(BASE_POI_NUM):
#    for j in range(len(grid_base_config)):
#        sum_ = 0
#        for k in range(REPEAT_TIME):
#            (district_sum_, district_num_) = all_err_list[j*REPEAT_TIME+k][i];
#            sum_ = sum_ + district_sum_/district_num_;
#        err_table[i, j] = sum_/REPEAT_TIME
#activate_map = np.argmin(err_table,1)       
##activate_map = np.zeros(66, dtype=np.int)
##activate_map = np.ones(66, dtype=np.int)


## mixture model performance checking -------------------------------------
#total_sum = 0
#total_num = 0
#district_err = []
#for i in range(len(data_list)):
##for i in range(0,1):
#    (my_data,my_label) = data_list[i]
#    train_num = int(np.floor(len(my_data)*TRAIN_RATIO))
#    valid_num = int(np.floor(len(my_data)*VALID_RATIO))
#    # test check!
#    test_data = my_data[train_num:valid_num]
#    test_label = my_label[train_num:valid_num]
##    test_data = my_data[valid_num:]
##    test_label = my_label[valid_num:]
#    
#    slot_num = test_data.shape[0]
#    res_distric_list = []
#    activated_model_idx = activate_map[i]
#    for j in range(test_data.shape[0]):
#        slot_test = test_data[j,:].reshape((1,test_data.shape[1]))
#        sum_ = 0
#        for k in range(REPEAT_TIME):
#            sum_ = sum_ + all_model_list[activated_model_idx*REPEAT_TIME+k][i].predict(slot_test)[0,0]
#        res_distric_list.append(sum_/REPEAT_TIME)
#        
#    (tmp_sum, tmp_num) = evaluate(np.asarray(res_distric_list),test_label)
#    total_sum = total_sum + tmp_sum
#    total_num = total_num + tmp_num
#    district_err.append(tmp_sum/tmp_num)
#    
#print 'final mape: %f\n' %(total_sum/total_num)

## predict ----------------------------------------------------------------
#OUTPUT_TEST_FEATURE_PATH_FILEPATH = '/Users/channerduan/Desktop/didi_test_data.csv'
#OUTPUT_RESULT_FILEPATH = '/Users/channerduan/Desktop/didi.csv'
#time_str = '2016-01-23-46,2016-01-23-58,2016-01-23-70,2016-01-23-82,2016-01-23-94,2016-01-23-106,2016-01-23-118,2016-01-23-130,2016-01-23-142,2016-01-25-58,2016-01-25-70,2016-01-25-82,2016-01-25-94,2016-01-25-106,2016-01-25-118,2016-01-25-130,2016-01-25-142,2016-01-27-46,2016-01-27-58,2016-01-27-70,2016-01-27-82,2016-01-27-94,2016-01-27-106,2016-01-27-118,2016-01-27-130,2016-01-27-142,2016-01-29-58,2016-01-29-70,2016-01-29-82,2016-01-29-94,2016-01-29-106,2016-01-29-118,2016-01-29-130,2016-01-29-142,2016-01-31-46,2016-01-31-58,2016-01-31-70,2016-01-31-82,2016-01-31-94,2016-01-31-106,2016-01-31-118,2016-01-31-130,2016-01-31-142'
#list_slot_str = time_str.split(',')
#test_data = genfromtxt(open(OUTPUT_TEST_FEATURE_PATH_FILEPATH,'r'), delimiter=',', dtype='int')
#slot_num = test_data.shape[0]
#f_out = open(OUTPUT_RESULT_FILEPATH,'w')
#res_list = []
#for i in range(BASE_POI_NUM):
#    res_distric_list = []
#    activated_model_idx = activate_map[i]
#    for j in range(test_data.shape[0]):
#        slot_test = test_data[j,:].reshape((1,test_data.shape[1]))
#        sum_ = 0
#        for k in range(REPEAT_TIME):
#            sum_ = sum_ + all_model_list[activated_model_idx*REPEAT_TIME+k][i].predict(slot_test)[0,0]
#        res_distric_list.append(sum_/REPEAT_TIME)
#    res_list.append(res_distric_list)
#for i in range(slot_num):
#    for j in range(BASE_POI_NUM):
#        f_out.write('%d,%s,%f\n' %(j+1,list_slot_str[i],res_list[j][i]))
#f_out.close()






