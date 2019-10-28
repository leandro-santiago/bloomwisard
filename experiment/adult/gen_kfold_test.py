import numpy as np
import math
import sys
from timeit import default_timer as timer
sys.path.append("../../") 

labels = {">50K": 0, ">50K.": 0, "<=50K": 1, "<=50K.": 1}
nominal_att = {1: {'Self-emp-inc': 6, 'State-gov': 0, 'Without-pay': 7, 'Private': 2, 'Local-gov': 4, 'Self-emp-not-inc': 1, 'Federal-gov': 3, 'Never-worked': 8, '?': 5}, 3: {'Masters':3, 'Prof-school': 10, '12th': 15, 'Assoc-voc': 7, '1st-4th': 13, 'Assoc-acdm': 6, 'HS-grad': 1, 'Bachelors': 0, '9th': 4, '5th-6th': 11, 'Some-college': 5, '11th': 2, '10th': 12, 'Doctorate': 9, 'Preschool': 14, '7th-8th': 8}, 5: {'Separated': 4, 'Widowed': 6, 'Divorced': 2, 'Married-spouse-absent': 3, 'Never-married': 0, 'Married-AF-spouse': 5, 'Married-civ-spouse': 1}, 6: {'Farming-fishing': 8, 'Armed-Forces': 13, 'Craft-repair': 6, 'Other-service': 4, 'Transport-moving': 7, 'Prof-specialty': 3, 'Sales': 5, 'Exec-managerial': 1, 'Handlers-cleaners': 2, '?': 11, 'Adm-clerical': 0, 'Protective-serv': 12, 'Tech-support': 10, 'Priv-house-serv': 14, 'Machine-op-inspct': 9}, 7: {'Own-child': 3, 'Wife': 2, 'Unmarried': 4, 'Other-relative': 5, 'Husband': 1, 'Not-in-family': 0}, 8: {'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'White': 0, 'Other': 4, 'Black': 1}, 9: {'Male': 0, 'Female': 1}, 13: {'Canada': 10, 'Hong': 38, 'Dominican-Republic': 24, 'Italy': 14, 'Ireland': 39, 'Outlying-US(Guam-USVI-etc)': 32, 'Scotland': 33, 'Cambodia': 17, 'France': 26, 'Peru': 31, 'Laos': 20, 'Ecuador': 19, 'Iran': 12, 'Cuba': 1, 'Guatemala': 27, 'Germany': 11, 'Thailand': 18, 'Haiti': 22, 'Poland': 15, '?': 4, 'Holand-Netherlands': 41, 'Philippines': 13, 'Vietnam': 37, 'Hungary': 40, 'England': 9, 'South': 6, 'Jamaica': 2, 'Honduras': 8, 'Portugal': 23, 'Mexico': 5, 'El-Salvador': 25, 'India': 3, 'Puerto-Rico': 7, 'China': 28, 'Yugoslavia': 30, 'United-States': 0, 'Trinadad&Tobago': 34, 'Greece': 35, 'Japan': 29, 'Taiwan': 21, 'Nicaragua': 36, 'Columbia': 16}}

base_path = "../../dataset/adult/"
data_name = "adult.data"

adult_data = []
adult_label = []

with open(base_path + data_name, "r") as fd:
    for line in fd:
        split_data = line.strip().split(",")
        adult_label.append(labels[split_data[-1].strip()])

        aux = []
        for i in range(len(split_data[:-1])):
            if nominal_att.has_key(i):
                aux.append(nominal_att[i][split_data[i].strip()])
            else:
                idata = int(split_data[i])
                aux.append(idata)
            
        adult_data.append(aux)
        
#Generating k-fold data
k = 10
total_size = len(adult_data)
fold_size = total_size / k
fold_size_last = fold_size + total_size % k

n_fold = k - 1
j1 = 0
j2 = 0

print adult_data
print adult_label
print total_size

for i in range(n_fold):
    start = j1
    with open(base_path + str(i) + "-fold-test.label", "w") as fd:    
        for l in range(fold_size):
            fd.write(str(adult_label[j1]) + "\n")
            j1 += 1

    end = j1

    with open(base_path + str(i) + "-fold-test.data", "w") as fd:    
        for l in range(fold_size):
            for d in adult_data[j2][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(adult_data[j2][-1]) + "\n")
            j2 += 1

    with open(base_path + str(i) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(adult_label[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(adult_label[l]) + "\n")

    with open(base_path + str(i) + "-fold-train.data", "w") as fd:    
        for l in range(start):
            for d in adult_data[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(adult_data[l][-1]) + "\n")

        for l in range(end, total_size):
            for d in adult_data[l][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(adult_data[l][-1]) + "\n")


#Last fold
with open(base_path + str(n_fold) + "-fold-test.label", "w") as fd:    
    for l in range(fold_size_last):
        fd.write(str(adult_label[j1]) + "\n")
        j1 += 1

with open(base_path + str(n_fold) + "-fold-train.label", "w") as fd:    
        for l in range(start):
            fd.write(str(adult_label[l]) + "\n")
            
        for l in range(end, total_size):
            fd.write(str(adult_label[l]) + "\n")

with open(base_path + str(n_fold) + "-fold-test.data", "w") as fd:    
    for l in range(fold_size_last):
        for d in adult_data[j2][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(adult_data[j2][-1]) + "\n")
        j2 += 1

with open(base_path + str(n_fold) + "-fold-train.data", "w") as fd:    
    for l in range(start):
        for d in adult_data[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(adult_data[l][-1]) + "\n")

    for l in range(end, total_size):
        for d in adult_data[l][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(adult_data[l][-1]) + "\n")


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
            fd.write(str(adult_label[indexes[j1]]) + "\n")
            j1 += 1

with open(base_path + "2b3-train.data", "w") as fd:    
    for i in range(t):
        for l in range(part_size):
            for d in adult_data[indexes[j2]][:-1]:
                fd.write(str(d) + ",")
            fd.write(str(adult_data[indexes[j2]][-1]) + "\n")
            j2 += 1

with open(base_path + "1b3-test.label", "w") as fd:    
    for l in range(part_size_last):
        fd.write(str(adult_label[indexes[j1]])  + "\n")
        j1 += 1

with open(base_path + "1b3-test.data", "w") as fd:    
    for l in range(part_size_last):
        for d in adult_data[indexes[j2]][:-1]:
            fd.write(str(d) + ",")
        fd.write(str(adult_data[indexes[j2]][-1]) + "\n")
        j2 += 1
