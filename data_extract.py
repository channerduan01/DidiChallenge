import os.path
import time
import numpy as np
from numpy import genfromtxt, savetxt
#
CLUSTER_MAP_FILEPATH = 'season_2/training_data/cluster_map/cluster_map'
#ORDER_PATH_DIRECTORY = 'season_2/test_set_1/order_data/'
ORDER_PATH_DIRECTORY = 'season_2/training_data/order_data/'

def getTimeIdx(timeStr):
    t = time.strptime(timeStr, "%Y-%m-%d %H:%M:%S")
    return int(t.tm_hour*6 + np.floor(t.tm_min/10))+1

# init poi cluster index
cluster_dict = {}
for item in genfromtxt(open(CLUSTER_MAP_FILEPATH,'r'), delimiter='\t', dtype='string'):
    cluster_dict[item[0]] = int(item[1])
BASE_POI_NUM = len(cluster_dict)

def readOneDayData(filename):
    data_table = np.zeros(shape=(BASE_POI_NUM*144, 4),dtype=np.float32)
    f = open(filename, 'r')
    line = f.readline()
    while line:
        tmplist = line.split('\t')
        tmplist[-1] = tmplist[-1][0:-1]
        for replace_idx in [3,4]:
            if cluster_dict.has_key(tmplist[replace_idx]):
                tmplist[replace_idx] = cluster_dict[tmplist[replace_idx]]
            else:
                tmplist[replace_idx] = 0
        tmplist[5] = float(tmplist[5])
        tmplist[6] = getTimeIdx(tmplist[6])
        
        idx = (tmplist[6]-1)*len(cluster_dict)+tmplist[3]-1
        data_table[idx,0] = tmplist[6]
        data_table[idx,1] = tmplist[3]
        data_table[idx,2] = data_table[idx,2]+1
        if tmplist[1] != 'NULL':
            data_table[idx,3] = data_table[idx,3]+1
        line = f.readline()
    f.close()
    return data_table
    
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

# process order data
feature_list = None
unsupervised_dataset = np.zeros((144*21,66), dtype=np.int)
unsupervised_idx = 0
if not 'feature_list' in dir() or feature_list == None:
    feature_list = []
    y_list = []
    for filename in os.listdir(ORDER_PATH_DIRECTORY):
        if filename.find('order_data') == -1:
           continue
        print 'loading: %s' %filename.split('_')[-1]
        data_table = readOneDayData(ORDER_PATH_DIRECTORY+filename)
        for i in range(144-3):
            feature_list_tmp = []
            feature_list_tmp.append(i+4)    # indicates the next time slot
            for j in range(3):
                s_idx = (i+j)*BASE_POI_NUM
                for idx_ in range(s_idx,s_idx+BASE_POI_NUM):
                    feature_list_tmp.append(int(data_table[idx_,2]-data_table[idx_,3]))
            y_list_tmp = []
            s_idx = (i+3)*BASE_POI_NUM
            for idx_ in range(s_idx,s_idx+BASE_POI_NUM):
                    y_list_tmp.append(int(data_table[idx_,2]-data_table[idx_,3]))
            feature_list.append(feature_list_tmp)
            y_list.append(y_list_tmp)
        unsupervised_dataset[unsupervised_idx*144:(unsupervised_idx+1)*144,:] \
            = (data_table[:,2]-data_table[:,3]).astype(np.int).reshape(144,66)
        unsupervised_idx = unsupervised_idx+1        
#        break # test mark
    # serial output
    data_ = np.asarray(feature_list)
    label_ = np.asarray(y_list)
    

    # shuffle output
#    shuffle_idx = range(len(feature_list))
#    np.random.shuffle(shuffle_idx)
#    data_ = np.zeros((len(feature_list),len(feature_list[0])), dtype=np.int)
#    label_ = np.zeros((len(y_list),len(y_list[0])), dtype=np.int)
#    for i in range(len(shuffle_idx)):
#        idx_ = shuffle_idx[i]
#        data_[i, :] = np.asarray(feature_list[idx_])
#        label_[i, :] = np.asarray(y_list[idx_])

OUTPUT_PATH_DIRECTORY = '/Users/channerduan/Desktop/'
savetxt(OUTPUT_PATH_DIRECTORY+'didi_train_data.csv', data_, delimiter=',', fmt='%d', comments='')
savetxt(OUTPUT_PATH_DIRECTORY+'didi_train_label.csv', label_, delimiter=',', fmt='%d', comments='')
savetxt(OUTPUT_PATH_DIRECTORY+'didi_unsupervised.csv', unsupervised_dataset, delimiter=',', fmt='%d', comments='')















