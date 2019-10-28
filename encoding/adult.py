import sys

nominal_att = {1: {'Self-emp-inc': 6, 'State-gov': 0, 'Without-pay': 7, 'Private': 2, 'Local-gov': 4, 'Self-emp-not-inc': 1, 'Federal-gov': 3, 'Never-worked': 8, '?': 5}, 3: {'Masters':3, 'Prof-school': 10, '12th': 15, 'Assoc-voc': 7, '1st-4th': 13, 'Assoc-acdm': 6, 'HS-grad': 1, 'Bachelors': 0, '9th': 4, '5th-6th': 11, 'Some-college': 5, '11th': 2, '10th': 12, 'Doctorate': 9, 'Preschool': 14, '7th-8th': 8}, 5: {'Separated': 4, 'Widowed': 6, 'Divorced': 2, 'Married-spouse-absent': 3, 'Never-married': 0, 'Married-AF-spouse': 5, 'Married-civ-spouse': 1}, 6: {'Farming-fishing': 8, 'Armed-Forces': 13, 'Craft-repair': 6, 'Other-service': 4, 'Transport-moving': 7, 'Prof-specialty': 3, 'Sales': 5, 'Exec-managerial': 1, 'Handlers-cleaners': 2, '?': 11, 'Adm-clerical': 0, 'Protective-serv': 12, 'Tech-support': 10, 'Priv-house-serv': 14, 'Machine-op-inspct': 9}, 7: {'Own-child': 3, 'Wife': 2, 'Unmarried': 4, 'Other-relative': 5, 'Husband': 1, 'Not-in-family': 0}, 8: {'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'White': 0, 'Other': 4, 'Black': 1}, 9: {'Male': 0, 'Female': 1}, 13: {'Canada': 10, 'Hong': 38, 'Dominican-Republic': 24, 'Italy': 14, 'Ireland': 39, 'Outlying-US(Guam-USVI-etc)': 32, 'Scotland': 33, 'Cambodia': 17, 'France': 26, 'Peru': 31, 'Laos': 20, 'Ecuador': 19, 'Iran': 12, 'Cuba': 1, 'Guatemala': 27, 'Germany': 11, 'Thailand': 18, 'Haiti': 22, 'Poland': 15, '?': 4, 'Holand-Netherlands': 41, 'Philippines': 13, 'Vietnam': 37, 'Hungary': 40, 'England': 9, 'South': 6, 'Jamaica': 2, 'Honduras': 8, 'Portugal': 23, 'Mexico': 5, 'El-Salvador': 25, 'India': 3, 'Puerto-Rico': 7, 'China': 28, 'Yugoslavia': 30, 'United-States': 0, 'Trinadad&Tobago': 34, 'Greece': 35, 'Japan': 29, 'Taiwan': 21, 'Nicaragua': 36, 'Columbia': 16}}

#for k in nominal_att:
#    print k, "-", len(nominal_att[k])

def load_data(base_path):
    data_min = None
    data_max = None
    train_data = []
    train_label = []

    test_data = []
    test_label = []
    
    with open(base_path + "adult.data", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []

            if data_min is None:
                data_min = {}
                data_max = {}

                for i in range(len(split_data[:-1])):
                    if not nominal_att.has_key(i):
                        data_min[i] = sys.maxint
                        data_max[i] = 0
               
            for i in range(len(split_data[:-1])):
                if nominal_att.has_key(i):
                    aux.append(nominal_att[i][split_data[i].strip()])
                else:
                    idata = int(split_data[i])

                    if idata > data_max[i]:
                        data_max[i] = idata

                    if idata < data_min[i]:
                        data_min[i] = idata

                    aux.append(idata)
    
            train_data.append(aux)

    with open(base_path + "train.label", "r") as fd:
        for line in fd:
            train_label.append(int(line.strip()))

    with open(base_path + "adult.test", "r") as fd:
        for line in fd:
            split_data = line.strip().split(",")
            aux = []
            for i in range(len(split_data[:-1])):
                if nominal_att.has_key(i):
                    aux.append(nominal_att[i][split_data[i].strip()])
                else:
                    idata = int(split_data[i])
                    aux.append(idata)

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
                data_min = {}
                data_max = {}

                for i in range(len(split_data)):
                    if not nominal_att.has_key(i):
                        data_min[i] = sys.maxint
                        data_max[i] = 0    
                    
            
            for i in range(len(split_data)):
                idata = int(split_data[i])
                
                if data_max.has_key(i):
                    if idata > data_max[i]:
                        data_max[i] = idata

                    if idata < data_min[i]:
                        data_min[i] = idata

                aux.append(idata)
                
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

    return train_data, train_label, test_data, test_label, data_min, data_max