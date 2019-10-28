import sys

nominal_att = [{"b" : 0, "c" : 1, "x" : 2, "f": 3, "k": 4, "s" : 5}, {"f": 0, "g": 1, "y": 2, "s" : 3}, {"n" : 0, "b" : 1, "c" : 2, "g": 3, "r": 4, "p" : 5, "u" : 6, "e" : 7, "w" : 8, "y" : 9}, {"t" : 0, "f": 1}, {"a" : 0, "l" : 1, "c" : 2, "y": 3, "f": 4, "m" : 5, "n" : 6, "p" : 7, "s" : 8}, {"a" : 0, "d" : 1, "f" : 2, "n": 3}, {"c" : 0, "w" : 1, "d" : 2}, {"b" : 0, "n" : 1}, {"k" : 0, "n" : 1, "b" : 2, "h": 3, "g": 4, "r" : 5, "o" : 6, "p" : 7, "u" : 8, "e" : 9, "w" : 10, "y" : 11}, {"e" : 0, "t" : 1}, {"b" : 0, "c" : 1, "u" : 2, "e": 3, "z": 4, "r" : 5, "?" : 6}, {"f": 0, "y": 1, "k": 2, "s" : 3}, {"f": 0, "y": 1, "k": 2, "s" : 3}, {"n": 0, "b": 1, "c": 2, "g" : 3, "o" : 4, "p" : 5, "e" : 6, "w" : 7, "y" : 8}, {"n": 0, "b": 1, "c": 2, "g" : 3, "o" : 4, "p" : 5, "e" : 6, "w" : 7, "y" : 8}, {"p": 0, "u": 1}, {"n": 0, "o": 1, "w": 2, "y" : 3}, {"n": 0, "o": 1, "t": 2}, {"c": 0, "e": 1, "f": 2, "l" : 3, "n" : 4, "p" : 5, "s" : 6, "z" : 7}, {"k": 0, "n": 1, "b": 2, "h" : 3, "r" : 4, "o" : 5, "u" : 6, "w" : 7, "y" : 8}, {"a": 0, "c": 1, "n": 2, "s" : 3, "v" : 4, "y" : 5}, {"g": 0, "l": 1, "m": 2, "p" : 3, "u" : 4, "w" : 5, "d" : 6}]

def load_3data(base_path):
    train_data = []
    train_label = []

    test_data = []
    test_label = []
    
    with open(base_path + "2b3-train.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []

            for s in split_data:
                aux.append(int(s))
                
            train_data.append(aux)

    with open(base_path + "2b3-train.label", "r") as fd:
        for line in fd:
            train_label.append(int(line.strip()))

    with open(base_path + "1b3-test.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []
            for s in split_data:
                aux.append(int(s))
            test_data.append(aux)

    with open(base_path + "1b3-test.label", "r") as fd:
        for line in fd:
            test_label.append(int(line.strip()))

    return train_data, train_label, test_data, test_label

def load_fold(base_path, k):
    train_data = []
    train_label = []

    test_data = []
    test_label = []
    
    with open(base_path + str(k) + "-fold-train.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []
            
            for s in split_data:
                aux.append(int(s))
                
            train_data.append(aux)

    with open(base_path + str(k) + "-fold-train.label", "r") as fd:
        for line in fd:
            train_label.append(int(line.strip()))

    with open(base_path + str(k) + "-fold-test.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []
            for s in split_data:
                aux.append(int(s))
            test_data.append(aux)

    with open(base_path + str(k) + "-fold-test.label", "r") as fd:
        for line in fd:
            test_label.append(int(line.strip()))

    return train_data, train_label, test_data, test_label