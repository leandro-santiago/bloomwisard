import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 
from core import wnn
from encoding import thermometer
from encoding import australian
from encoding import hamming_code

#Load Australian data
base_path = "../../dataset/australian/"

#K-fold
folds_train_bin = []
folds_test_bin = []
folds_train_label = []
folds_test_label = []
folds_thermomethers = []
k = 10

bits_encoding = 20
nominal_size = {0: 2, 3: 3, 4: 14, 5: 9, 7:2, 8:2, 10: 2, 11: 3}
#nominal_length = {0: 8, 3: 8, 4: 16, 5: 16, 7:8, 8:8, 10: 8, 11: 8}
nominal_length = {0: 16, 3: 16, 4: 32, 5: 32, 7:16, 8:16, 10: 16, 11: 16}
nominal_length2 = {0: 5, 3 : 5, 4: 5, 5: 5, 7: 5, 8:5, 10: 5, 11: 5}

for i in range(k):
    train_data, train_label, test_data, test_label, data_min, data_max = australian.load_fold(base_path, i)
    folds_train_label.append(train_label)
    folds_test_label.append(test_label)
    
    #print data_min, data_max
    folds_thermomethers.append({})

    for key in data_max:
        folds_thermomethers[i][key] = thermometer.Thermometer(data_min[key], data_max[key], bits_encoding)
    
    folds_train_bin.append([])
    folds_test_bin.append([])

    j = 0
    for data in train_data:
        folds_train_bin[i].append([])
        folds_train_bin[i][j].append(np.array([], dtype=bool))
        
        for a in range(len(data)):
            if folds_thermomethers[i].has_key(a):
                binarr = folds_thermomethers[i][a].binarize(data[a])
                #print "C ", binarr
            else:
                #binarr = hamming_code.get_code(data[a], nominal_length[a])
                p = data[a] * nominal_length2[a]
                binarr = np.zeros(nominal_length2[a]*nominal_size[a], dtype=bool)
                for b in range(nominal_length2[a]):
                    binarr[p] = 1
                    p += 1

            folds_train_bin[i][j] = np.append(folds_train_bin[i][j], binarr)  
        
        j += 1

    j = 0
    for data in test_data:
        folds_test_bin[i].append([])
        folds_test_bin[i][j].append(np.array([], dtype=bool))

        for a in range(len(data)):
            if folds_thermomethers[i].has_key(a):
                binarr = folds_thermomethers[i][a].binarize(data[a])
                #print "C ", binarr
            else:
                #binarr = hamming_code.get_code(data[a], nominal_length[a])
                p = data[a] * nominal_length2[a]
                binarr = np.zeros(nominal_length2[a]*nominal_size[a], dtype=bool)
                for b in range(nominal_length2[a]):
                    binarr[p] = 1
                    p += 1
  
            folds_test_bin[i][j] = np.append(folds_test_bin[i][j], binarr)  
  
        j += 1


#Wisard
num_classes = 2
tuple_list = [2, 4, 8, 14, 16, 18, 20, 22, 24, 26, 28, 30]
btuple_list = [2, 4, 8, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40, 56]

test_length = len(folds_test_bin[0])
entry_size = len(folds_train_bin[0][0])

acc_list = [0.0] * len(tuple_list)
dacc_list = [0.0] * len(tuple_list)
bacc_list = [0.0] * len(btuple_list)

#Classifying each fold
for f in range(k):
    #Wisard
    j = 0
    for t in tuple_list:
        wisard = wnn.Wisard(entry_size, t, num_classes)
        wisard.train(folds_train_bin[f], folds_train_label[f])
        rank_result = wisard.rank(folds_test_bin[f])    
        
        num_hits = 0

        for i in range(test_length):
            #if rank_result[i] == folds_test_label[f][i]:
            if not (rank_result[i] ^ test_label[i]):
                num_hits += 1

        acc_list[j] += (float(num_hits)/float(test_length))
        j += 1

    #Dic Wisard
    j = 0
    for t in tuple_list:
        dwisard = wnn.DictWisard(entry_size, t, num_classes)
        dwisard.train(folds_train_bin[f], folds_train_label[f])
        rank_result = dwisard.rank(folds_test_bin[f])    
        
        num_hits = 0

        for i in range(test_length):
            #if rank_result[i] == folds_test_label[f][i]:
            if not (rank_result[i] ^ test_label[i]):
                num_hits += 1

        dacc_list[j] += (float(num_hits)/float(test_length))
        j += 1


    #Bloom Wisard
    j = 0
    for t in btuple_list:
        bwisard = wnn.BloomWisard(entry_size, t, num_classes, len(folds_train_label[f]))
        bwisard.train(folds_train_bin[f], folds_train_label[f])
        rank_result = bwisard.rank(folds_test_bin[f])    
        
        num_hits = 0

        for i in range(test_length):
            #if rank_result[i] == folds_test_label[f][i]:
            if not (rank_result[i] ^ test_label[i]):
                num_hits += 1

        bacc_list[j] += (float(num_hits)/float(test_length))
        j += 1



for i in range(len(tuple_list)):
    acc_list[i] /= k

for i in range(len(tuple_list)):
    dacc_list[i] /= k

for i in range(len(btuple_list)):
    bacc_list[i] /= k

print "Tuples=", tuple_list
print "Wisard Accuracy=", acc_list
print "Dict Wisard Accuracy=", dacc_list
print "Tuples=", btuple_list
print "BloomWisard Accuracy=",bacc_list


