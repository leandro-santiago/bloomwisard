import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 
from encoding import iris

#Load Iris data
base_path = "../../dataset/iris/"
data_name = "../../dataset/iris/iris.data"
iris_data = iris.load(data_name)

iris_x = iris_data.data
iris_y = iris_data.target

#Generating k-fold data
total_size = len(iris_x)
k = 10
fold_size = total_size / k
fold_size_last = fold_size + total_size % k

n_fold = k - 1
j1 = 0
j2 = 0

#print str(iris_x[0])

for i in range(n_fold):
    start = j1
    with open(base_path + str(i) + "-fold-test.label", "w") as fd:    
        for l in range(fold_size):
            fd.write(str(iris_y[j1]) + "\n")
            j1 += 1
    end = j1

    with open(base_path + str(i) + "-fold-test.data", "w") as fd:    
        for l in range(fold_size):
            for d in iris_x[j2][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(iris_x[j2][-1]) + "\n")
            j2 += 1

    with open(base_path + str(i) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(iris_y[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(iris_y[l]) + "\n")
    
    with open(base_path + str(i) + "-fold-train.data", "w") as fd:    
        for l in range(start):
            for d in iris_x[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(iris_x[l][-1]) + "\n")

        for l in range(end, total_size):
            for d in iris_x[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(iris_x[l][-1]) + "\n")

#Last fold
with open(base_path + str(n_fold) + "-fold-test.label", "w") as fd:    
    for l in range(fold_size_last):
        fd.write(str(iris_y[j1]) + "\n")
        j1 += 1

with open(base_path + str(n_fold) + "-fold-train.label", "w") as fd:    
    for l in range(start):
        fd.write(str(iris_y[l]) + "\n")
        
    for l in range(end, total_size):
        fd.write(str(iris_y[l]) + "\n")
        
with open(base_path + str(n_fold) + "-fold-test.data", "w") as fd:    
    for l in range(fold_size_last):
        for d in iris_x[j2][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(iris_x[j2][-1]) + "\n")
        j2 += 1

with open(base_path + str(n_fold) + "-fold-train.data", "w") as fd:    
    for l in range(start):
        for d in iris_x[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(iris_x[l][-1]) + "\n")

    for l in range(end, total_size):
        for d in iris_x[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(iris_x[l][-1]) + "\n")

#Split data in traning and testing input
s = 3
part_size = len(iris_x) / s
part_size_last = part_size + len(iris_x) % s

np.random.seed(100)
indexes = np.random.permutation(len(iris_x))

t = s - 1
j1 = 0
j2 = 0
with open(base_path + "2b3-train.label", "w") as fd:    
    for i in range(t):
        for l in range(part_size):
            fd.write(str(iris_y[indexes[j1]]) + "\n")
            j1 += 1

with open(base_path + "2b3-train.data", "w") as fd:    
    for i in range(t):
        for l in range(part_size):
            for d in iris_x[indexes[j2]][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(iris_x[indexes[j2]][-1]) + "\n")
            j2 += 1

with open(base_path + "1b3-test.label", "w") as fd:    
    for l in range(part_size_last):
        fd.write(str(iris_y[indexes[j1]])  + "\n")
        j1 += 1

with open(base_path + "1b3-test.data", "w") as fd:    
    for l in range(part_size_last):
        for d in iris_x[indexes[j2]][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(iris_x[indexes[j2]][-1]) + "\n")
        j2 += 1
