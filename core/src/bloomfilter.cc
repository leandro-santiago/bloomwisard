#include <iostream>
#include <math.h>
#include <stdint.h>
#include "bitarray.h"
#include "murmur3.h"

using namespace std;

class BloomFilter {
public:
    BloomFilter(long int capacity, float error):
    capacity(capacity), error(error)
    {
        numBits = (-capacity * (log(error) / pow(log(2), 2))) + 1;
        numHashes = (int)((numBits * (log(2)/capacity)) + 1);

        bitmemory = bitarray_new(numBits);
        seed = 100;
    }

    BloomFilter(long int capacity, float error, int numHashes, long int numBits):
    capacity(capacity), error(error)
    {
        if (numBits <= 0) {
            this->numBits = (-capacity * (log(error) / pow(log(2), 2))) + 1;
        } else {
            this->numBits = numBits;
        }

        if (numHashes <= 0) {
            this->numHashes = (this->numBits * (log(2)/capacity)) + 1;    
        }    
        
        bitmemory = bitarray_new(this->numBits);
        seed = 100;
    }

    ~BloomFilter()
    {
        bitarray_free(bitmemory);
    }

    void add(bitarray_t * data)
    {
        int num_bytes = (data->num_bits >> 3) + ((data->num_bits & 0x00000007) > 0);
        
        uint64_t mm3_val[2];
        MurmurHash3_x64_128(data->bitarray, num_bytes, seed, &mm3_val);
        uint64_t index;
        int i;
        uint32_t i1, i2;
        
        //Double Hashing
        for (i = 0; i < numHashes; i++) {
            index = (mm3_val[0] + i * mm3_val[1]) % numBits;
            i1 = index >> 6;
            i2 = index & 0x3F;
            bitmemory->bitarray[i1] |= (1UL << i2);
        }

    }

    int lookup(bitarray_t * data)
    {
        int res = 1;
        int num_bytes = (data->num_bits >> 3) + ((data->num_bits & 0x00000007) > 0);
        
        uint64_t mm3_val[2];
        MurmurHash3_x64_128(data->bitarray, num_bytes, seed, &mm3_val);
        uint64_t index;
        int i;
        uint32_t i1, i2;

        //Double Hashing
        for (i = 0; i < numHashes; i++) {
            index = (mm3_val[0] + i * mm3_val[1]) % numBits;
            i1 = index >> 6;
            i2 = index & 0x3F;
            res &= (bitmemory->bitarray[i1] & (1UL << i2)) >> i2;
        }

        return res;
    }

    void info() 
    {

        cout << "Bit Memory = " << numBits << " bits, capacity = " << capacity << ", error = " << error << ", Num Hashes = " << numHashes << endl;
        
        /*
        int j;

        for (j = 0; j < bitmemory->bitarray_size; j++) {
            cout << bitmemory->bitarray[j] << ", ";
        }
        cout << endl;*/
    }

    void reset()
    {
        int i;

        for (i = 0; i < bitmemory->bitarray_size; i++) {
            bitmemory->bitarray[i] = 0;
        }
    }

    long int calculate_num_bits()
    {   
        return (-capacity * (log(error) / pow(log(2), 2))) + 1;
    }

    long int calculate_num_hashes()
    {
        long int nbits = (-capacity * (log(error) / pow(log(2), 2))) + 1;
        return (nbits * (log(2)/capacity)) + 1;
    }

    long int getNumBits()
    {
        return numBits;
    }

    long int getCapacity()
    {
        return capacity;
    }

    float getError()
    {
        return error;
    }

    int getNumHashes()
    {
        return numHashes;
    }

private:
    long int capacity;
	float error;
	long int numBits;
	int numHashes;
	bitarray_t * bitmemory;
    int seed;
};

/*int main() {    
    bitarray_t * a = bitarray_new(10);

    BloomFilter b(1000, 0.1);
    b.add(a);
    int r = b.lookup(a);

    free(a);
    
    cout << "DOne " << r << endl;
    return 0;
}*/