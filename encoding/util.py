import sys

def load_data(base_path):
    data_min = None
    data_max = None
    train_data = []
    train_label = []

    test_data = []
    test_label = []
    
    with open(base_path + "train.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []

            if data_min is None:
                data_min = []

                for s in split_data:
                    data_min.append(sys.maxint)

            if data_max is None:
                data_max = []

                for s in split_data:
                    data_max.append(0)    
            i = 0
            for s in split_data:
                fdata = float(s)

                if fdata > data_max[i]:
                    data_max[i] = fdata

                if fdata < data_min[i]:
                    data_min[i] = fdata

                aux.append(fdata)
                i += 1
            train_data.append(aux)

    with open(base_path + "train.label", "r") as fd:
        for line in fd:
            train_label.append(int(line.strip()))

    with open(base_path + "test.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []
            for s in split_data:
                aux.append(float(s))
            test_data.append(aux)

    with open(base_path + "test.label", "r") as fd:
        for line in fd:
            test_label.append(int(line.strip()))

    return train_data, train_label, test_data, test_label, data_min, data_max

def load_3data(base_path):
    #data_min = sys.maxint
    #data_max = 0
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
                data_min = []

                for s in split_data:
                    data_min.append(sys.maxint)

            if data_max is None:
                data_max = []

                for s in split_data:
                    data_max.append(0)    
            i = 0
            for s in split_data:
                fdata = float(s)

                if fdata > data_max[i]:
                    data_max[i] = fdata

                if fdata < data_min[i]:
                    data_min[i] = fdata

                aux.append(fdata)
                i += 1
            train_data.append(aux)

    with open(base_path + "2b3-train.label", "r") as fd:
        for line in fd:
            train_label.append(int(line.strip()))

    with open(base_path + "1b3-test.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []
            for s in split_data:
                aux.append(float(s))
            test_data.append(aux)

    with open(base_path + "1b3-test.label", "r") as fd:
        for line in fd:
            test_label.append(int(line.strip()))

    return train_data, train_label, test_data, test_label, data_min, data_max

def load_fold(base_path, k):
    #data_min = sys.maxint
    #data_max = 0
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
                data_min = []

                for s in split_data:
                    data_min.append(sys.maxint)

            if data_max is None:
                data_max = []

                for s in split_data:
                    data_max.append(0) 
            i = 0
            for s in split_data:
                fdata = float(s)

                if fdata > data_max[i]:
                    data_max[i] = fdata

                if fdata < data_min[i]:
                    data_min[i] = fdata

                aux.append(fdata)
                i += 1
            train_data.append(aux)

    with open(base_path + str(k) + "-fold-train.label", "r") as fd:
        for line in fd:
            train_label.append(int(line.strip()))

    with open(base_path + str(k) + "-fold-test.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []
            for s in split_data:
                aux.append(float(s))
            test_data.append(aux)

    with open(base_path + str(k) + "-fold-test.label", "r") as fd:
        for line in fd:
            test_label.append(int(line.strip()))

    return train_data, train_label, test_data, test_label, data_min, data_max