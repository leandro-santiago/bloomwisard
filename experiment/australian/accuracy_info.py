import numpy as np
import sys
from timeit import default_timer as timer
sys.path.append("../../") 
from core import wnn
from encoding import thermometer
from encoding import australian
from encoding import hamming_code

#Load Wine data
base_path = "../../dataset/australian/"

#2/3 Test
bits_encoding = 20
train_data, train_label, test_data, test_label, data_min, data_max = australian.load_3data(base_path)

ths = {}

for index in data_max:
    ths[index] = thermometer.Thermometer(data_min[index], data_max[index], bits_encoding)

train_bin = []
test_bin = []

nominal_size = {0: 2, 3: 3, 4: 14, 5: 9, 7:2, 8:2, 10: 2, 11: 3}
nominal_length = {0: 8, 3: 8, 4: 16, 5: 16, 7:8, 8:8, 10: 8, 11: 8}
nominal_length2 = {0: 5, 3 : 5, 4: 5, 5: 5, 7: 5, 8:5, 10: 5, 11: 5}

i = 0
for data in train_data:
    train_bin.append(np.array([], dtype=bool))

    for a in range(len(data)):
        if ths.has_key(a):
            binarr = ths[a].binarize(data[a])
            #print "C ", binarr
        else:
            binarr = hamming_code.get_code(data[a], nominal_length[a])
            
            #p = data[a] * nominal_length2[a]
            #binarr = np.zeros(nominal_length2[a]*nominal_size[a], dtype=bool)
            #print a, ":", nominal_length2[a], "*", data[a], "=", p, " - ", nominal_size[a]
           
            #for b in range(nominal_length2[a]):
            #    binarr[p] = 1
            #    p += 1

        train_bin[i] = np.append(train_bin[i], binarr)  
        #print "N ", data[a], ":", binarr

    i += 1


i = 0
for data in test_data:
    test_bin.append(np.array([], dtype=bool))
    
    for a in range(len(data)):
        if ths.has_key(a):
            binarr = ths[a].binarize(data[a])
            #print "C ", binarr
        else:
            binarr = hamming_code.get_code(data[a], nominal_length[a])
            #p = data[a] * nominal_length2[a]
            #binarr = np.zeros(nominal_length2[a]*nominal_size[a], dtype=bool)
            #for b in range(nominal_length2[a]):
            #    binarr[p] = 1
            #    p += 1    
        test_bin[i] = np.append(test_bin[i], binarr)  
    
    i += 1

#print test_label
#Wisard
num_classes = 2
tuple_list = [2, 4, 8, 14, 16, 18, 20, 22, 24, 26, 28, 30]
acc_list = []
test_length = len(test_label)
entry_size = len(train_bin[0])

#print entry_size

for t in tuple_list:
    wisard = wnn.Wisard(entry_size, t, num_classes)
    wisard.train(train_bin, train_label)
    rank_result = wisard.rank(test_bin)    
    
    num_hits = 0

    for i in range(test_length):
        if not (rank_result[i] ^ test_label[i]):
            num_hits += 1

    acc_list.append(float(num_hits)/float(test_length))

#Bloom Wisard
btuple_list = [2, 4, 8, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40, 56]
bacc_list = []

for t in btuple_list:
    bwisard = wnn.BloomWisard(entry_size, t, num_classes, len(train_bin))
    bwisard.train(train_bin, train_label)
    rank_result = bwisard.rank(test_bin)    
    
    num_hits = 0

    for i in range(test_length):
        #if rank_result[i] == test_label[i]:
        if not (rank_result[i] ^ test_label[i]):
            num_hits += 1

    bacc_list.append(float(num_hits)/float(test_length))

print "Tuples=", tuple_list
print "Wisard Accuracy=", acc_list
print "Tuples=", btuple_list
print "BloomWisard Accuracy=",bacc_list

