import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 

base_path = "../../dataset/shuttle/"
train_data_name = "shuttle.trn"
test_data_name = "shuttle.tst"

train_data = []
train_label = []

test_data = []
test_label = []

with open(base_path + train_data_name, "r") as fd:
    for line in fd:
        split_data = line.strip().split(" ")
        train_label.append(int(split_data[-1].strip()) - 1)

        aux = []
        for s in split_data[:-1]:
            aux.append(float(s))

        train_data.append(aux)


with open(base_path + test_data_name, "r") as fd:
    for line in fd:
        split_data = line.strip().split(" ")
        test_label.append(int(split_data[-1].strip()) - 1)

        aux = []
        for s in split_data[:-1]:
            aux.append(float(s))

        test_data.append(aux)

with open(base_path + "train.label", "w") as fd:    
    for d in train_label:
        fd.write(str(d) + "\n")
        
with open(base_path + "train.data", "w") as fd:    
    for d in train_data:
        for a in d[:-1]:
            fd.write(str(a) + ",")
        fd.write(str(d[-1]) + "\n")
        
with open(base_path + "test.label", "w") as fd:    
    for d in test_label:
        fd.write(str(d) + "\n")
        
with open(base_path + "test.data", "w") as fd:    
    for d in test_data:
        for a in d[:-1]:
            fd.write(str(a) + ",")
        fd.write(str(d[-1]) + "\n")
