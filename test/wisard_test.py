import numpy as np
import sys
sys.path.append("../") 
from core import wnn

a = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=bool)
b = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)

#Discriminator
disc = wnn.Discriminator(20, 4)
disc.train(a)

print disc.rank(a), disc.rank(b)

#BloomDiscriminator
bloom_disc = wnn.BloomDiscriminator(20, 4, 1000)
bloom_disc.train(a)

print bloom_disc.rank(a), bloom_disc.rank(b)
bloom_disc.info()

#Wisard
print "Wisard"
wisard2 = wnn.Wisard(20, 4, 2)

c = [a, b]
#t = np.ndarray(shape=(2, 20), buffer=np.array(c, dtype=bool), dtype=bool)

wisard2.train(c, [0, 1])
#print t
print wisard2.rank(c)

wisard2.info()

#BloomWisard
print "Bloom Wisard"
bwisard = wnn.BloomWisard(20, 4, 2, 1000)
bwisard.train(c, [0, 1])
print bwisard.rank(c)

bwisard.info()

print "Dict Wisard"
wisard = wnn.DictWisard(20, 4, 2)
wisard.train(c, [0, 1])
#print t
print wisard.rank(c)
wisard.info()

print "Done"
