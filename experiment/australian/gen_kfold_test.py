import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 

base_path = "../../dataset/australian/"
data_name = "australian.dat"

australian_data = []
australian_label = []
continuos = {1:0, 2:0, 6:0, 9:0, 12:0, 13:0}
minus_one = {3:0, 4:0, 5:0, 11:0}

with open(base_path + data_name, "r") as fd:
    for line in fd:
        split_data = line.strip().split(" ")
        australian_label.append(int(split_data[-1]))

        aux = []
        for i in range(len(split_data[:-1])):
            if continuos.has_key(i):
                aux.append(float(split_data[i]))
            else:
                if minus_one.has_key(i):
                    aux.append(int(split_data[i]) - 1)
                else:
                    aux.append(int(split_data[i]))

        australian_data.append(aux)

#Generating k-fold data
k = 10
total_size = len(australian_data)
fold_size = total_size / k
fold_size_last = fold_size + total_size % k

n_fold = k - 1
j1 = 0
j2 = 0

print australian_data
print australian_label
print total_size

for i in range(n_fold):
    start = j1
    with open(base_path + str(i) + "-fold-test.label", "w") as fd:    
        for l in range(fold_size):
            fd.write(str(australian_label[j1]) + "\n")
            j1 += 1

    end = j1

    with open(base_path + str(i) + "-fold-test.data", "w") as fd:    
        for l in range(fold_size):
            for d in australian_data[j2][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(australian_data[j2][-1]) + "\n")
            j2 += 1

    with open(base_path + str(i) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(australian_label[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(australian_label[l]) + "\n")

    with open(base_path + str(i) + "-fold-train.data", "w") as fd:    
        for l in range(start):
            for d in australian_data[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(australian_data[l][-1]) + "\n")

        for l in range(end, total_size):
            for d in australian_data[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(australian_data[l][-1]) + "\n")


#Last fold
with open(base_path + str(n_fold) + "-fold-test.label", "w") as fd:    
    for l in range(fold_size_last):
        fd.write(str(australian_label[j1]) + "\n")
        j1 += 1

with open(base_path + str(n_fold) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(australian_label[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(australian_label[l]) + "\n")

with open(base_path + str(n_fold) + "-fold-test.data", "w") as fd:    
    for l in range(fold_size_last):
        for d in australian_data[j2][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(australian_data[j2][-1]) + "\n")
        j2 += 1

with open(base_path + str(n_fold) + "-fold-train.data", "w") as fd:    
    for l in range(start):
        for d in australian_data[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(australian_data[l][-1]) + "\n")

    for l in range(end, total_size):
        for d in australian_data[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(australian_data[l][-1]) + "\n")


#Split data in traning and testing input
s = 3
part_size = total_size / s
part_size_last = part_size + total_size % s

np.random.seed(100)
indexes = np.random.permutation(total_size)

t = s - 1
j1 = 0
j2 = 0

with open(base_path + "2b3-train.label", "w") as fd:    
    for i in range(t):
        for l in range(part_size):
            fd.write(str(australian_label[indexes[j1]]) + "\n")
            j1 += 1

with open(base_path + "2b3-train.data", "w") as fd:    
    for i in range(t):
        for l in range(part_size):
            for d in australian_data[indexes[j2]][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(australian_data[indexes[j2]][-1]) + "\n")
            j2 += 1

with open(base_path + "1b3-test.label", "w") as fd:    
    for l in range(part_size_last):
        fd.write(str(australian_label[indexes[j1]])  + "\n")
        j1 += 1

with open(base_path + "1b3-test.data", "w") as fd:    
    for l in range(part_size_last):
        for d in australian_data[indexes[j2]][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(australian_data[indexes[j2]][-1]) + "\n")
        j2 += 1
