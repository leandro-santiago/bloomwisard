import numpy as np
import struct

def read_idx(filename):
    with open(filename, "rb") as fd:
        #Magic Number
        zero, data_type, dims = struct.unpack('>HBB', fd.read(4))
        shape = tuple(struct.unpack('>I', fd.read(4))[0] for d in range(dims))
        return np.fromstring(fd.read(), dtype=np.uint8).reshape(shape)

def binarize_array(array):
    mean = np.mean(array)

    return (array>=mean)*1
