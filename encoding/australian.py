import sys

continuos = {1:0, 2:0, 6:0, 9:0, 12:0, 13:0}

def load_3data(base_path):
    data_min = None
    data_max = None
    train_data = []
    train_label = []

    test_data = []
    test_label = []   
    
    with open(base_path + "2b3-train.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []

            if data_min is None:
                data_min = {}
                data_max = {}

                for i in range(len(split_data)):
                    if continuos.has_key(i):
                        data_min[i] = sys.maxint
                        data_max[i] = 0

            for i in range(len(split_data)):
                if continuos.has_key(i):
                    fdata = float(split_data[i])

                    if fdata > data_max[i]:
                        data_max[i] = fdata

                    if fdata < data_min[i]:
                        data_min[i] = fdata
                else:
                    fdata = int(split_data[i])

                aux.append(fdata)
                
            train_data.append(aux)

    with open(base_path + "2b3-train.label", "r") as fd:
        for line in fd:
            train_label.append(int(line.strip()))

    with open(base_path + "1b3-test.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []

            for i in range(len(split_data)):
                if continuos.has_key(i):
                    fdata = float(split_data[i])
                else:
                    fdata = int(split_data[i])

                aux.append(fdata)
            
            test_data.append(aux)

    with open(base_path + "1b3-test.label", "r") as fd:
        for line in fd:
            test_label.append(int(line.strip()))

    return train_data, train_label, test_data, test_label, data_min, data_max

def load_fold(base_path, k):
    data_min = None
    data_max = None
    train_data = []
    train_label = []

    test_data = []
    test_label = []
    
    with open(base_path + str(k) + "-fold-train.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []
            
            if data_min is None:
                data_min = {}
                data_max = {}

                for i in range(len(split_data)):
                    if continuos.has_key(i):
                        data_min[i] = sys.maxint
                        data_max[i] = 0

            for i in range(len(split_data)):
                if continuos.has_key(i):
                    fdata = float(split_data[i])

                    if fdata > data_max[i]:
                        data_max[i] = fdata

                    if fdata < data_min[i]:
                        data_min[i] = fdata
                else:
                    fdata = int(split_data[i])
                aux.append(fdata)

            train_data.append(aux)

    with open(base_path + str(k) + "-fold-train.label", "r") as fd:
        for line in fd:
            train_label.append(int(line.strip()))

    with open(base_path + str(k) + "-fold-test.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []

            for i in range(len(split_data)):
                if continuos.has_key(i):
                    fdata = float(split_data[i])
                else:
                    fdata = int(split_data[i])

                aux.append(fdata)
                
            test_data.append(aux)

    with open(base_path + str(k) + "-fold-test.label", "r") as fd:
        for line in fd:
            test_label.append(int(line.strip()))

    return train_data, train_label, test_data, test_label, data_min, data_max