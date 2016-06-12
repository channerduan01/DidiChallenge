# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:28:09 2016

@author: channerduan
"""
import matplotlib.pyplot as plt  
import numpy as np  
from numpy import genfromtxt, savetxt


CORRECT_RATE_LSTM_FILEPATH = 'data/correct_rate_from_lstm'
CORRECT_RATE_LSTM_FILEPATH2 = 'data/correct_rate_from_lstm2'

CORRECT_RATE_SVM_FILEPATH = 'data/correct_rate_from_svm'

tmp_str = genfromtxt(open(CORRECT_RATE_LSTM_FILEPATH,'r'), delimiter=' ', dtype='string')
correct_rate_from_lstm = tmp_str[:,3].astype(np.float)

tmp_str = genfromtxt(open(CORRECT_RATE_LSTM_FILEPATH2,'r'), delimiter=' ', dtype='string')
correct_rate_from_lstm2 = tmp_str[:,3].astype(np.float)
    
tmp_str = genfromtxt(open(CORRECT_RATE_SVM_FILEPATH,'r'), delimiter=' ', dtype='string')
correct_rate_from_svm = tmp_str[:,1].astype(np.float)
    

#whole_err_table = np.c_[err_table,correct_rate_from_lstm,correct_rate_from_svm,np.asarray(district_err)]
#plt.figure(figsize=(20,10))
#plt.plot(whole_err_table)
#plt.legend(['ann_linear','ann_sigmoid','lstm','svm','ann_matrix'])
#plt.title('Error Comparison')




whole_err_table = np.c_[correct_rate_from_lstm,correct_rate_from_lstm2]
plt.figure(figsize=(20,10))
plt.plot(whole_err_table)
plt.legend(['lstm','lstm2'])
plt.title('Error Comparison')