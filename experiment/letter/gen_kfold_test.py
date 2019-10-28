import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 

base_path = "../../dataset/letter/"
data_name = "letter-recognition.data"

labels = {"A": 0, "B": 1, "C": 2, "D": 3, "E" : 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25}

letter_data = []
letter_label = []

with open(base_path + data_name, "r") as fd:
    for line in fd:
        split_data = line.strip().split(",")
        letter_label.append(labels[split_data[0].strip()])

        aux = []
        for s in split_data[1:]:
            aux.append(float(s))

        letter_data.append(aux)

#Generating k-fold data
k = 10
total_size = len(letter_data)
fold_size = total_size / k
fold_size_last = fold_size + total_size % k

n_fold = k - 1
j1 = 0
j2 = 0

print letter_data
print letter_label
print total_size

for i in range(n_fold):
    start = j1
    with open(base_path + str(i) + "-fold-test.label", "w") as fd:    
        for l in range(fold_size):
            fd.write(str(letter_label[j1]) + "\n")
            j1 += 1

    end = j1

    with open(base_path + str(i) + "-fold-test.data", "w") as fd:    
        for l in range(fold_size):
            for d in letter_data[j2][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(letter_data[j2][-1]) + "\n")
            j2 += 1

    with open(base_path + str(i) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(letter_label[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(letter_label[l]) + "\n")

    with open(base_path + str(i) + "-fold-train.data", "w") as fd:    
        for l in range(start):
            for d in letter_data[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(letter_data[l][-1]) + "\n")

        for l in range(end, total_size):
            for d in letter_data[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(letter_data[l][-1]) + "\n")


#Last fold
with open(base_path + str(n_fold) + "-fold-test.label", "w") as fd:    
    for l in range(fold_size_last):
        fd.write(str(letter_label[j1]) + "\n")
        j1 += 1

with open(base_path + str(n_fold) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(letter_label[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(letter_label[l]) + "\n")

with open(base_path + str(n_fold) + "-fold-test.data", "w") as fd:    
    for l in range(fold_size_last):
        for d in letter_data[j2][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(letter_data[j2][-1]) + "\n")
        j2 += 1

with open(base_path + str(n_fold) + "-fold-train.data", "w") as fd:    
    for l in range(start):
        for d in letter_data[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(letter_data[l][-1]) + "\n")

    for l in range(end, total_size):
        for d in letter_data[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(letter_data[l][-1]) + "\n")


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
            fd.write(str(letter_label[indexes[j1]]) + "\n")
            j1 += 1

with open(base_path + "2b3-train.data", "w") as fd:    
    for i in range(t):
        for l in range(part_size):
            for d in letter_data[indexes[j2]][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(letter_data[indexes[j2]][-1]) + "\n")
            j2 += 1

with open(base_path + "1b3-test.label", "w") as fd:    
    for l in range(part_size_last):
        fd.write(str(letter_label[indexes[j1]])  + "\n")
        j1 += 1

with open(base_path + "1b3-test.data", "w") as fd:    
    for l in range(part_size_last):
        for d in letter_data[indexes[j2]][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(letter_data[indexes[j2]][-1]) + "\n")
        j2 += 1
