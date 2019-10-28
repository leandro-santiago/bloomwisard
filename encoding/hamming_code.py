import numpy as np
import math

#Generating matrix for Hamming Code with dmin = 3
G_3 = np.matrix('1 1 1 0 0 0 0; 1 0 0 1 1 0 0; 0 1 0 1 0 1 0; 1 1 0 1 0 0 1')

#Generating matrix for Hamming Code with dmin = 4
G_4 = np.matrix('1 1 1 0 0 0 0 1; 1 0 0 1 1 0 0 1; 0 1 0 1 0 1 0 1; 1 1 0 1 0 0 1 0')

#Hamming Code has dmin = 4
trivial_db = []

for i in range(16):
    code = np.matrix(np.fromstring(("{0:04b}").format(i), np.int8) - 48)
    trivial_db.append((code * G_4) % 2)

#arr = np.array([], dtype=bool)
#a = trivial_db[1].astype(bool)
#print np.append(arr, trivial_db[1])

def get_code(index, length=0):
    code = np.array([], dtype=bool)

    if length > 0:
        bits = length>>1
        c = bits
        aux = index

        while (c > 0):
            i = aux & 0x7
            aux >>= 4    
            c -= 4
            #print i, trivial_db[i]
            code = np.append(code, trivial_db[i])
        
    else:
        if index == 0:
            code = np.append(code, trivial_db[0])
        else:
            bits = int(math.floor(math.log(index, 2))) + 1
            c = bits
            aux = index

            while (c > 0):
                i = aux & 0x7
                aux >>= 4    
                c -= 4
                #print i, trivial_db[i]
                code = np.append(code, trivial_db[i])
            
    return code
