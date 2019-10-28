import numpy as np
import math
import time
import sys
sys.path.append("../../") 
from core import wnn
from encoding import mnist

#Mnist data filename
data_path = "../../dataset/mnist/"
train_data_filename = "train-images.idx3-ubyte"
train_label_filename = "train-labels.idx1-ubyte"
test_data_filename = "t10k-images.idx3-ubyte"
test_label_filename = "t10k-labels.idx1-ubyte"

#Read mnist dataset
train_data = mnist.read_idx(data_path + train_data_filename)
train_label = mnist.read_idx(data_path + train_label_filename)
test_data = mnist.read_idx(data_path + test_data_filename)
test_label = mnist.read_idx(data_path + test_label_filename)

#print train_data

train_label = train_label.astype(int)
test_label = test_label.astype(int)

#Convert to binary format
train_bin = []
test_bin = []
entry_size = train_data[0].shape[0] * train_data[0].shape[1]

#Binarize train data
for data in train_data:
    bin_data = mnist.binarize_array(data)
    bin_data = bin_data.astype(bool)
    bin_data = bin_data.reshape(entry_size,)
    train_bin.append(bin_data)

#Binarize test data
for data in test_data:
    bin_data = mnist.binarize_array(data)
    bin_data = bin_data.astype(bool)
    bin_data = bin_data.reshape(entry_size,)
    test_bin.append(bin_data)

train_bin = np.asarray(train_bin)
test_bin = np.asarray(test_bin)

#Wisard
num_classes = 10
test_length = len(test_label)

#Bloom Wisard
btuple_list = [2, 4, 7, 8, 14, 16, 28, 49, 56]
bacc_list = []

#for t in btuple_list:
bwisard = wnn.BloomWisard(entry_size, 28, num_classes, 50000, error=0.5)
bwisard.train(train_bin, train_label)
rank_result = bwisard.rank(test_bin)    

bwisard_stats = bwisard.stats()
print bwisard_stats
num_hits = 0

for i in range(test_length):
    if rank_result[i] == test_label[i]:
        num_hits += 1

bacc_list.append(float(num_hits)/float(test_length))

#print "Tuples=", btuple_list
print "BloomWisard Accuracy=",bacc_list
