import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 

base_path = "../../dataset/mushroom/"
data_name = "agaricus-lepiota.data"

mushroom_data = []
mushroom_label = []

nominal_att = [{"b" : 0, "c" : 1, "x" : 2, "f": 3, "k": 4, "s" : 5}, {"f": 0, "g": 1, "y": 2, "s" : 3}, {"n" : 0, "b" : 1, "c" : 2, "g": 3, "r": 4, "p" : 5, "u" : 6, "e" : 7, "w" : 8, "y" : 9}, {"t" : 0, "f": 1}, {"a" : 0, "l" : 1, "c" : 2, "y": 3, "f": 4, "m" : 5, "n" : 6, "p" : 7, "s" : 8}, {"a" : 0, "d" : 1, "f" : 2, "n": 3}, {"c" : 0, "w" : 1, "d" : 2}, {"b" : 0, "n" : 1}, {"k" : 0, "n" : 1, "b" : 2, "h": 3, "g": 4, "r" : 5, "o" : 6, "p" : 7, "u" : 8, "e" : 9, "w" : 10, "y" : 11}, {"e" : 0, "t" : 1}, {"b" : 0, "c" : 1, "u" : 2, "e": 3, "z": 4, "r" : 5, "?" : 6}, {"f": 0, "y": 1, "k": 2, "s" : 3}, {"f": 0, "y": 1, "k": 2, "s" : 3}, {"n": 0, "b": 1, "c": 2, "g" : 3, "o" : 4, "p" : 5, "e" : 6, "w" : 7, "y" : 8}, {"n": 0, "b": 1, "c": 2, "g" : 3, "o" : 4, "p" : 5, "e" : 6, "w" : 7, "y" : 8}, {"p": 0, "u": 1}, {"n": 0, "o": 1, "w": 2, "y" : 3}, {"n": 0, "o": 1, "t": 2}, {"c": 0, "e": 1, "f": 2, "l" : 3, "n" : 4, "p" : 5, "s" : 6, "z" : 7}, {"k": 0, "n": 1, "b": 2, "h" : 3, "r" : 4, "o" : 5, "u" : 6, "w" : 7, "y" : 8}, {"a": 0, "c": 1, "n": 2, "s" : 3, "v" : 4, "y" : 5}, {"g": 0, "l": 1, "m": 2, "p" : 3, "u" : 4, "w" : 5, "d" : 6}]

with open(base_path + data_name, "r") as fd:
    for line in fd:
        split_data = line.strip().split(",")

        if split_data[0].strip() == 'e':
            l = 0
        else:
            l = 1
        mushroom_label.append(l)

        aux = []
        for i in range(len(split_data[1:])):
            index = i + 1    
            aux.append(nominal_att[i][split_data[index].strip()])

        mushroom_data.append(aux)

#Generating k-fold data
k = 10
total_size = len(mushroom_data)
fold_size = total_size / k
fold_size_last = fold_size + total_size % k

n_fold = k - 1
j1 = 0
j2 = 0

print mushroom_data
print mushroom_label
print total_size

for i in range(n_fold):
    start = j1
    with open(base_path + str(i) + "-fold-test.label", "w") as fd:    
        for l in range(fold_size):
            fd.write(str(mushroom_label[j1]) + "\n")
            j1 += 1

    end = j1

    with open(base_path + str(i) + "-fold-test.data", "w") as fd:    
        for l in range(fold_size):
            for d in mushroom_data[j2][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(mushroom_data[j2][-1]) + "\n")
            j2 += 1

    with open(base_path + str(i) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(mushroom_label[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(mushroom_label[l]) + "\n")

    with open(base_path + str(i) + "-fold-train.data", "w") as fd:    
        for l in range(start):
            for d in mushroom_data[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(mushroom_data[l][-1]) + "\n")

        for l in range(end, total_size):
            for d in mushroom_data[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(mushroom_data[l][-1]) + "\n")


#Last fold
with open(base_path + str(n_fold) + "-fold-test.label", "w") as fd:    
    for l in range(fold_size_last):
        fd.write(str(mushroom_label[j1]) + "\n")
        j1 += 1

with open(base_path + str(n_fold) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(mushroom_label[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(mushroom_label[l]) + "\n")

with open(base_path + str(n_fold) + "-fold-test.data", "w") as fd:    
    for l in range(fold_size_last):
        for d in mushroom_data[j2][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(mushroom_data[j2][-1]) + "\n")
        j2 += 1

with open(base_path + str(n_fold) + "-fold-train.data", "w") as fd:    
    for l in range(start):
        for d in mushroom_data[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(mushroom_data[l][-1]) + "\n")

    for l in range(end, total_size):
        for d in mushroom_data[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(mushroom_data[l][-1]) + "\n")


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
            fd.write(str(mushroom_label[indexes[j1]]) + "\n")
            j1 += 1

with open(base_path + "2b3-train.data", "w") as fd:    
    for i in range(t):
        for l in range(part_size):
            for d in mushroom_data[indexes[j2]][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(mushroom_data[indexes[j2]][-1]) + "\n")
            j2 += 1

with open(base_path + "1b3-test.label", "w") as fd:    
    for l in range(part_size_last):
        fd.write(str(mushroom_label[indexes[j1]])  + "\n")
        j1 += 1

with open(base_path + "1b3-test.data", "w") as fd:    
    for l in range(part_size_last):
        for d in mushroom_data[indexes[j2]][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(mushroom_data[indexes[j2]][-1]) + "\n")
        j2 += 1
