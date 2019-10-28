def isNum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


base_path = "../../dataset/adult/"

labels = {">50K": 0, ">50K.": 0, "<=50K": 1, "<=50K.": 1}

nominal_att = {}
index_att = {}

train_label = []
test_label = []

with open(base_path + "adult.data", "r") as fd:
    for line in fd: 
        split_data = line.strip().split(",")
        train_label.append(labels[split_data[-1].strip()])

        i = 0
        for f in split_data[:-1]:
            if not isNum(f.strip()):
                if nominal_att.has_key(i):
                    if not nominal_att[i].has_key(f.strip()):
                        nominal_att[i][f.strip()] = index_att[i]
                        index_att[i] += 1
                else:
                    nominal_att[i] = {f.strip(): 0}
                    index_att[i] = 1
                
            #print i, isNum(f.strip())
            i += 1

print nominal_att

with open(base_path + "adult.test", "r") as fd:
    for line in fd: 
        split_data = line.strip().split(",")
        test_label.append(labels[split_data[-1].strip()])


with open(base_path + "train.label", "w") as fd:
    for l in train_label:
        fd.write(str(l) + "\n")

with open(base_path + "test.label", "w") as fd:
    for l in test_label:
        fd.write(str(l) + "\n")