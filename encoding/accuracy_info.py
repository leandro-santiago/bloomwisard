import numpy as np
import sys
from timeit import default_timer as timer
sys.path.append("../../") 
from core import wnn
from encoding import thermometer
from encoding import adult

#Load Adult data
base_path = "../../dataset/adult/"
train_data, train_label, test_data, test_label, data_min, data_max = adult.load_data(base_path)


exit()
bits_encoding = 20
ths = []

for i in range(len(data_max)):
    ths.append(thermometer.Thermometer(data_min[i], data_max[i], bits_encoding))

train_bin = []
test_bin = []

i = 0
for data in train_data:
    train_bin.append(np.array([], dtype=bool))
    t = 0
    for v in data:
        binarr = ths[t].binarize(v)
        train_bin[i] = np.append(train_bin[i], binarr)  
        t += 1
    i += 1


i = 0
for data in test_data:
    test_bin.append(np.array([], dtype=bool))
    t = 0
    for v in data:
        binarr = ths[t].binarize(v)
        test_bin[i] = np.append(test_bin[i], binarr)  
        t += 1
    i += 1

#print test_label
#Wisard
num_classes = 3
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
        if rank_result[i] == test_label[i]:
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
        if rank_result[i] == test_label[i]:
            num_hits += 1

    bacc_list.append(float(num_hits)/float(test_length))

print "Tuples=", tuple_list
print "Wisard Accuracy=", acc_list
print "Tuples=", btuple_list
print "BloomWisard Accuracy=",bacc_list

