import os
import numpy as np
import re
import sys
from bitarray import bitarray
from thermometer import Thermometer

class Iris:
    def __init__(self):
        self.DESCR = "Iris Plants Database"
        self.data = []
        self.feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
        self.target = []
        self.target_names = ["setosa", "versicolor", "virginica"]


    def encode(self, row):
        column = row.strip().split(",")
        self.data.append(np.array(column[:-1], dtype=np.float64))
        self.target.append(self.target_names.index(column[-1][5:]))

def load(filename, operation = None):
    basename = os.path.basename(filename)
    cfiledata = basename + ".data"
    cfiletarget = basename + ".target"

    if operation == "u" or not (os.path.isfile(cfiledata + ".npy") or os.path.isfile(cfiletarget + ".npy")): #Update .npy files
        fp = open(filename, "r")
        iris = Iris()

        try:
            for line in fp:
                if re.search(r"\d", line):
                    iris.encode(line)


            iris.data = np.ndarray(shape=(len(iris.data), 4), buffer=np.array(iris.data, dtype=np.float64), dtype=np.float64)
            iris.target = np.ndarray(shape=(len(iris.target),), buffer=np.array(iris.target, dtype=np.int64), dtype=np.int64)
        finally:
            fp.close()
            np.save(cfiledata, iris.data)
            np.save(cfiletarget, iris.target)
    else:
        iris = Iris()
        iris.data = np.load(cfiledata + ".npy")
        iris.target = np.load(cfiletarget + ".npy")

    return iris

def encoding(input, num_bits = 8):
    inputs = []
    th = Thermometer(input.data.min(), input.data.max(), num_bits)

    for i in range(len(input.data)):
        bit_array = bitarray(0)
        row = input.data[i]

        for elem in row:
             bit_array.extend(th.code(elem))

        inputs.append({"class" : input.target[i], "bitarray": bit_array})


    return inputs

