import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 
from core import wnn
from encoding import thermometer
from encoding import util

#Load Segment data
base_path = "../../dataset/segment/"

#2/3 Test
bits_encoding = 20
train_data, train_label, test_data, test_label, data_min, data_max = util.load_3data(base_path)

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

#K-fold
folds_train_bin = []
folds_test_bin = []
folds_train_label = []
folds_test_label = []
folds_thermomethers = []
k = 10

for i in range(k):
    aux_train_data, aux_train_label, aux_test_data, aux_test_label, data_min, data_max = util.load_fold(base_path, i)
    folds_train_label.append(aux_train_label)
    folds_test_label.append(aux_test_label)
    
    #print data_min, data_max
    folds_thermomethers.append([])

    for t in range(len(data_max)):
        folds_thermomethers[i].append(thermometer.Thermometer(data_min[t], data_max[t], bits_encoding))

    folds_train_bin.append([])
    folds_test_bin.append([])

    j = 0
    for data in aux_train_data:
        folds_train_bin[i].append([])
        folds_train_bin[i][j].append(np.array([], dtype=bool))
        t = 0
        for v in data:
            binarr = folds_thermomethers[i][t].binarize(v)
            folds_train_bin[i][j] = np.append(folds_train_bin[i][j], binarr)  
            t += 1
        j += 1

    j = 0
    for data in aux_test_data:
        folds_test_bin[i].append([])
        folds_test_bin[i][j].append(np.array([], dtype=bool))
        t = 0
        for v in data:
            binarr = folds_thermomethers[i][t].binarize(v)
            folds_test_bin[i][j] = np.append(folds_test_bin[i][j], binarr)  
            t += 1
        j += 1


#Parameters
num_classes = 7
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

#K-fold cross validation ---------------------------------------------------------
#Wisard

test_length = len(folds_test_label[0])
kf_training_time = []
kf_testing_time = []
kf_wacc_list = []

for r in range(num_runs):
    for f in range(k):
        wisard = wnn.Wisard(entry_size, tuple_bit, num_classes)
        
        #Training
        start = timer()
        wisard.train(folds_train_bin[f], folds_train_label[f])
        kf_training_time.append(timer() - start)
        
        #Testing
        start = timer()
        rank_result = wisard.rank(folds_test_bin[f])    
        kf_testing_time.append(timer() - start)

        #Accuracy
        num_hits = 0

        for i in range(test_length):
            if rank_result[i] == folds_test_label[f][i]:
                num_hits += 1

        kf_wacc_list.append(float(num_hits)/float(test_length))

kf_wisard_stats = wisard.stats()
del wisard

#DictWisard
kf_dtraining_time = []
kf_dtesting_time = []
kf_dacc_list = []
for r in range(num_runs):
    for f in range(k):
        dwisard = wnn.DictWisard(entry_size, tuple_bit, num_classes)

        #Training
        start = timer()
        dwisard.train(folds_train_bin[f], folds_train_label[f])
        kf_dtraining_time.append(timer() - start)
        
        #Testing
        start = timer()
        rank_result = dwisard.rank(folds_test_bin[f])    
        kf_dtesting_time.append(timer() - start)

        #Accuracy
        num_hits = 0

        for i in range(test_length):
            #if rank_result[i] == folds_test_label[f][i]:
            if not (rank_result[i] ^ folds_test_label[f][i]):
                num_hits += 1

        kf_dacc_list.append(float(num_hits)/float(test_length))
kf_dwisard_stats = dwisard.stats()
del dwisard

#Bloom Wisard
#capacity2 = len(folds_train_label[0])
capacity2 = 100
error = 0.1
kf_btraining_time = []
kf_btesting_time = []
kf_bacc_list = []

kb_stats = []
kb_training_time = [] 
kb_testing_time = [] 
kb_acc = []
kb_error = []

for e in range(len(errors)):
    kf_btraining_time = []
    kf_btesting_time = []
    kf_bacc_list = []

    for r in range(num_runs):
        for f in range(k):
            bwisard = wnn.BloomWisard(entry_size, tuple_bit, num_classes, capacity2, error=errors[e])
        
            #Training
            start = timer()
            bwisard.train(folds_train_bin[f], folds_train_label[f])
            kf_btraining_time.append(timer() - start)

            #Testing
            start = timer()
            rank_result = bwisard.rank(folds_test_bin[f])    
            kf_btesting_time.append(timer() - start)
            
            #Accuracy
            num_hits = 0

            for i in range(test_length):
                if rank_result[i] == folds_test_label[f][i]:
                    num_hits += 1

            kf_bacc_list.append(float(num_hits)/float(test_length))

    kb_training_time.append(kf_btraining_time)
    kb_testing_time.append(kf_btesting_time)
    kb_acc.append(kf_bacc_list)
    kb_stats.append(bwisard.stats())
    kb_error.append(bwisard.error())
#kf_bwisard_stats = bwisard.stats()
#kf_berror = bwisard.error()
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

    out.write("Wisard-10fold;" + str(entry_size) + ";" + str(tuple_bit) + ";" + str(kf_wisard_stats[0]) + ";;;;" + str(kf_wisard_stats[1]) + ";" + str(num_classes) + ";" + str(kf_wisard_stats[3]) + ";")
    out.write(str(np.mean(kf_wacc_list)) + ";" + str(np.std(kf_wacc_list)) + ";" + str(np.mean(kf_training_time)) + ";" + str(np.std(kf_training_time)) + ";" + str(np.mean(kf_testing_time)) + ";" + str(np.std(kf_testing_time)) + ";" + str(num_runs) + ";\n")

    out.write("Dict Wisard-10fold;" + str(entry_size) + ";" + str(tuple_bit) + ";" + str(kf_dwisard_stats[0]) + ";;;;" + str(kf_dwisard_stats[1]) + ";" + str(num_classes) + ";" + str(kf_dwisard_stats[2]) + ";")
    out.write(str(np.mean(kf_dacc_list)) + ";" + str(np.std(kf_dacc_list)) + ";" + str(np.mean(kf_dtraining_time)) + ";" + str(np.std(kf_dtraining_time)) + ";" + str(np.mean(kf_dtesting_time)) + ";" + str(np.std(kf_dtesting_time)) + ";" + str(num_runs) + ";\n")

    for i in range(len(errors)):
        out.write("Bloom Wisard-10fold;" + str(entry_size) + ";" + str(tuple_bit) + ";" + str(kb_stats[i][0]) + ";" + str(capacity2) + ";" + str(kb_error[i]) + ";" + str(kb_stats[i][4]) + ";" + str(kb_stats[i][1]) + ";" + str(num_classes) + ";" + str(kb_stats[i][3]) + ";")
        out.write(str(np.mean(kb_acc[i])) + ";" + str(np.std(kb_acc[i])) + ";" + str(np.mean(kb_training_time[i])) + ";" + str(np.std(kb_training_time[i])) + ";" + str(np.mean(kb_testing_time[i])) + ";" + str(np.std(kb_testing_time[i])) + ";" + str(num_runs) + ";\n")
