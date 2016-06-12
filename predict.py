import os.path
import time
import numpy as np
from numpy import genfromtxt, savetxt
#
CLUSTER_MAP_FILEPATH = 'season_2/test_set_2/cluster_map/cluster_map'
ORDER_PATH_DIRECTORY = 'season_2/test_set_2/order_data/'
OUTPUT_TEST_FEATURE_PATH_FILEPATH = '/Users/channerduan/Desktop/didi_test_data.csv'
OUTPUT_RESULT_FILEPATH = '/Users/channerduan/Desktop/didi.csv'

def getTimeIdx(timeStr):
    t = time.strptime(timeStr, "%Y-%m-%d %H:%M:%S")
    return int(t.tm_hour*6 + np.floor(t.tm_min/10))+1

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
    
if not os.path.exists(OUTPUT_TEST_FEATURE_PATH_FILEPATH):
    cluster_dict = {}
    for item in genfromtxt(open(CLUSTER_MAP_FILEPATH,'r'), delimiter='\t', dtype='string'):
        cluster_dict[item[0]] = int(item[1])
    BASE_POI_NUM = len(cluster_dict)
    feature_list = []
    for filename in os.listdir(ORDER_PATH_DIRECTORY):
        if filename.find('order_data') == -1:
           continue
        date_str = filename.split('_')[-2]
        data_table = readOneDayData(ORDER_PATH_DIRECTORY+filename)
        for i in range(144-3):
            s_idx = i*BASE_POI_NUM
            e_idx = (i+3)*BASE_POI_NUM
            if (data_table[s_idx,0] != 0 and data_table[e_idx-1,0] != 0):
                print 'predict slot:%d' %(i+4)
                feature_list_tmp = []
                feature_list_tmp.append(i+4)    # indicates the next time slot
                for j in range(3):
                    s_idx_tmp = (i+j)*BASE_POI_NUM
                    for idx_ in range(s_idx_tmp,s_idx_tmp+BASE_POI_NUM):
                        feature_list_tmp.append(int(data_table[idx_,2]-data_table[idx_,3]))
                feature_list.append(feature_list_tmp)
    #            predict_ = clf.predict(np.reshape(np.asarray(feature_list_tmp),(1,len(feature_list_tmp))))
    #            for j in range(BASE_POI_NUM):
    #                f_out.write('%d,%s-%d,%f\n' %(j+1,date_str,i+4,predict_[0,j]))
        break
    savetxt(OUTPUT_TEST_FEATURE_PATH_FILEPATH, np.asarray(feature_list), delimiter=',', fmt='%d', comments='')






