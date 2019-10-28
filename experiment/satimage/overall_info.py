import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 
from core import wnn
from encoding import thermometer
from encoding import util

#Load Shuttle data
base_path = "../../dataset/satimage/"

#Test
bits_encoding = 20
train_data, train_label, test_data, test_label, data_min, data_max = util.load_data(base_path)

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

#Parameters
num_classes = 6
tuple_bit = 20
test_length = len(test_label)
num_runs = 20

acc_list = []
training_time = []
testing_time = []

dacc_list = []
dtraining_time = []
dtesting_time = []

bacc_list = []
btraining_time = []
btesting_time = []
entry_size = len(train_bin[0])

#Wisard
for r in range(num_runs):
    wisard = wnn.Wisard(entry_size, tuple_bit, num_classes)

    #Training
    start = timer()
    wisard.train(train_bin, train_label)
    training_time.append(timer() - start)

    #Testing
    start = timer()
    rank_result = wisard.rank(test_bin)    
    testing_time.append(timer() - start)

    #Accuracy
    num_hits = 0

    for i in range(test_length):
        if rank_result[i] == test_label[i]:
            num_hits += 1

    acc_list.append(float(num_hits)/float(test_length))

wisard_stats = wisard.stats()
del wisard

#DictWisard
for r in range(num_runs):
    dwisard = wnn.DictWisard(entry_size, tuple_bit, num_classes)

    #Training
    start = timer()
    dwisard.train(train_bin, train_label)
    dtraining_time.append(timer() - start)
    
    #Testing
    start = timer()
    rank_result = dwisard.rank(test_bin)    
    dtesting_time.append(timer() - start)

    #Accuracy
    num_hits = 0

    for i in range(test_length):
        #if rank_result[i] == test_label[i]:
        if not (rank_result[i] ^ test_label[i]):
            num_hits += 1

    dacc_list.append(float(num_hits)/float(test_length))
dwisard_stats = dwisard.stats()
del dwisard


#Bloom Wisard
#capacity = len(train_label)
capacity = 100
error = 0.1
errors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

b_stats = []
b_training_time = [] 
b_testing_time = [] 
b_acc = []
b_error = []

for e in range(len(errors)):
    btraining_time = []
    btesting_time = []
    bacc_list = []

    for r in range(num_runs):
        bwisard = wnn.BloomWisard(entry_size, tuple_bit, num_classes, capacity, error=errors[e])

        #Training
        start = timer()
        bwisard.train(train_bin, train_label)
        btraining_time.append(timer() - start)

        #Testing
        start = timer()
        rank_result = bwisard.rank(test_bin)    
        btesting_time.append(timer() - start)
        
        #Accuracy
        num_hits = 0

        for i in range(test_length):
            if rank_result[i] == test_label[i]:
                num_hits += 1

        bacc_list.append(float(num_hits)/float(test_length))

    b_training_time.append(btraining_time)
    b_testing_time.append(btesting_time)
    b_acc.append(bacc_list)
    b_stats.append(bwisard.stats())
    b_error.append(bwisard.error())
#bwisard_stats = bwisard.stats()
#berror = bwisard.error()
del bwisard


#Writing output file
with open("stats.csv", "w") as out:
    out.write("WNN; Entry size; Tuple size; # Rams; Capacity; Error; # Hashes; Ram size; # Discriminators; Total Bits; Acc(%); Acc Std; Training(s); Training Std; Testing(s); Testing Std; Runs;\n")
    out.write("Wisard;" + str(entry_size) + ";" + str(tuple_bit) + ";" + str(wisard_stats[0]) + ";;;;" + str(wisard_stats[1]) + ";" + str(num_classes) + ";" + str(wisard_stats[3]) + ";")
    out.write(str(np.mean(acc_list)) + ";" + str(np.std(acc_list)) + ";" + str(np.mean(training_time)) + ";" + str(np.std(training_time)) + ";" + str(np.mean(testing_time)) + ";" + str(np.std(testing_time)) + ";" + str(num_runs) + ";\n")

    out.write("Dict Wisard;" + str(entry_size) + ";" + str(tuple_bit) + ";" + str(dwisard_stats[0]) + ";;;;" + str(dwisard_stats[1]) + ";" + str(num_classes) + ";" + str(dwisard_stats[2]) + ";")
    out.write(str(np.mean(dacc_list)) + ";" + str(np.std(dacc_list)) + ";" + str(np.mean(dtraining_time)) + ";" + str(np.std(dtraining_time)) + ";" + str(np.mean(dtesting_time)) + ";" + str(np.std(dtesting_time)) + ";" + str(num_runs) + ";\n")

    for i in range(len(errors)):
        out.write("Bloom Wisard;" + str(entry_size) + ";" + str(tuple_bit) + ";" + str(b_stats[i][0]) + ";" + str(capacity) + ";" + str(b_error[i]) + ";" + str(b_stats[i][4]) + ";" + str(b_stats[i][1]) + ";" + str(num_classes) + ";" + str(b_stats[i][3]) + ";")
        out.write(str(np.mean(b_acc[i])) + ";" + str(np.std(b_acc[i])) + ";" + str(np.mean(b_training_time[i])) + ";" + str(np.std(b_training_time[i])) + ";" + str(np.mean(b_testing_time[i])) + ";" + str(np.std(b_testing_time[i])) + ";" + str(num_runs) + ";\n")
