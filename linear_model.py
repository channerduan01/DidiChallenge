import matplotlib.pyplot as plt  
import numpy as np  
import scipy as sp  
from scipy.stats import norm  
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn import linear_model  
  
TRAIN_RATIO = 0.8
train_num = int(np.floor(len(data_)*TRAIN_RATIO))
train_data = data_[0:train_num]
test_data = data_[train_num:]
train_label = label_[0:train_num]
test_label = label_[train_num:]




#clf = Pipeline([('poly', PolynomialFeatures(degree=1)), 
#                ('linear', LinearRegression(fit_intercept=False))])

#clf = linear_model.LinearRegression()
#clf = linear_model.Ridge (alpha = 100000)     
clf = linear_model.Lasso(alpha = 10)      
clf.fit(train_data, train_label)
predict_result = clf.predict(test_data)
a = np.abs(test_label-predict_result)
b = test_label-predict_result
sum_ = 0
num_ = 0
for i in range(predict_result.shape[0]):
    for j in range(predict_result.shape[1]):
        if (test_label[i,j] != 0):
            sum_ = sum_ + a[i,j]/test_label[i,j]
            num_ = num_ + 1
print 'eva: %f, ls: %f\n' %(sum_/num_, np.sqrt(np.sum(b*b))/(predict_result.shape[0]*predict_result.shape[1]))







