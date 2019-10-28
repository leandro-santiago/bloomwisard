#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include "bloomfilter.cc"

namespace py = pybind11;

class BloomDiscriminator{
public:
    BloomDiscriminator(int entrySize, int tupleSize, long int capacity, py::kwargs kwargs) :
    entrySize(entrySize), tupleSize(tupleSize)
    {
        numRams = entrySize/tupleSize + ((entrySize%tupleSize) > 0);
        tuplesMapping = (int *)calloc(entrySize, sizeof(int));
        float error = 0.01;
        int numHash = 0;
        long int bloomSize = 0;

        for(auto arg: kwargs){
            if (string(py::str(arg.first)).compare("error") == 0) {
                error = arg.second.cast<float>();;
            }

            if (string(py::str(arg.first)).compare("nhash") == 0) {
                numHash = arg.second.cast<int>();;
            }

            if (string(py::str(arg.first)).compare("bloomsize") == 0) {
                bloomSize = arg.second.cast<long int>();;
            }

        }
        
        //Generating pseudo-random mapping
        int i;

        for (i = 0; i < entrySize; i++) {
            tuplesMapping[i] = i;
        }
        
        std::shuffle(tuplesMapping, tuplesMapping + entrySize, std::default_random_engine(std::random_device{}()));

        //Allocate Bloom Ram Memory
        bloomRams = vector<BloomFilter *>(numRams);
        
        for (i = 0; i < numRams; i++) {
            bloomRams[i] = new BloomFilter(capacity, error, numHash, bloomSize);
        }

        //Initialize number of bitarray for address position
        numIntAddr = (tupleSize >> 6); //divide by 64 bits
	    numIntAddr += ((tupleSize & 0x3F) > 0); //ceil quotient. If remainder > 0 then sum by 1
        addr = {tupleSize, numIntAddr, NULL};
        addr.bitarray = (uint64_t *) calloc(numIntAddr, sizeof(uint64_t));
    }

    BloomDiscriminator(int entrySize, int tupleSize, long int capacity, float error, int numHash, long int bloomSize) :
    entrySize(entrySize), tupleSize(tupleSize)
    {
        numRams = entrySize/tupleSize + ((entrySize%tupleSize) > 0);
        tuplesMapping = (int *)calloc(entrySize, sizeof(int));
        
        //Generating pseudo-random mapping
        int i;

        for (i = 0; i < entrySize; i++) {
            tuplesMapping[i] = i;
        }
        
        std::shuffle(tuplesMapping, tuplesMapping + entrySize, std::default_random_engine(std::random_device{}()));

        //Allocate Bloom Ram Memory
        bloomRams = vector<BloomFilter *>(numRams);
        
        for (i = 0; i < numRams; i++) {
            bloomRams[i] = new BloomFilter(capacity, error, numHash, bloomSize);
        }

        //Initialize number of bitarray for address position
        numIntAddr = (tupleSize >> 6); //divide by 64 bits
	    numIntAddr += ((tupleSize & 0x3F) > 0); //ceil quotient. If remainder > 0 then sum by 1
        addr = {tupleSize, numIntAddr, NULL};
        addr.bitarray = (uint64_t *) calloc(numIntAddr, sizeof(uint64_t));
    }

    ~BloomDiscriminator()
    {
        free(tuplesMapping);
        free(addr.bitarray);    

        unsigned int i;
        for (i = 0; i < bloomRams.size(); i ++) {
            delete bloomRams[i];
        }    
    }

    void info()
    {
        int i;
        long int totalBits = 0;
        cout << "Entry = " << entrySize << ", Tuples = " << tupleSize << ", RAMs = " << numRams << ", ";

        bloomRams[0]->info();

        for (i = 0; i < numRams; i++) {
            //cout << "Bloom RAM " << i << ": ";
            totalBits += bloomRams[i]->getNumBits();
        }

        cout << "Total Bits = " << totalBits << endl;
    }

    void train(const vector<bool>& data)
    {
        int i, j, k = 0, i1, i2;
        int addr_pos;

        for (i = 0; i < numRams; i++) {
            addr_pos = 0;
            
            for (j = 0; j < numIntAddr; j++) {
                addr.bitarray[j] = 0;
            }

            for (j = 0; (j < tupleSize) && (k < entrySize); j++) {
                i1 = addr_pos >> 6;//Divide by 64 to find the bitarray id
                i2 = addr_pos & 0x3F;//Obtain remainder to access the bitarray position
            
                addr.bitarray[i1] |= (data[tuplesMapping[k]] << i2);
                addr_pos++;
                k++;
            }

            bloomRams[i]->add(&addr); 
        }
    }

    int rank(const vector<bool>& data)
    {
        int rank = 0;
        int i, j, k = 0, i1, i2;
        int addr_pos;
        
        for (i = 0; i < numRams; i++) {
            addr_pos = 0;
            
            for (j = 0; j < numIntAddr; j++) {
                addr.bitarray[j] = 0;
            }

            for (j = 0; (j < tupleSize) && (k < entrySize); j++) {
                i1 = addr_pos >> 6;//Divide by 64 to find the bitarray id
                i2 = addr_pos & 0x3F;//Obtain remainder to access the bitarray position
            
                addr.bitarray[i1] |= (data[tuplesMapping[k]] << i2);
                addr_pos++;
                k++;
            }

            rank += bloomRams[i]->lookup(&addr); 
        }

        return rank;
    }

    void reset()
    {
        //Generating pseudo-random mapping
        int i;

        for (i = 0; i < entrySize; i++) {
            tuplesMapping[i] = i;
        }
        
        std::shuffle(tuplesMapping, tuplesMapping + entrySize, std::default_random_engine(std::random_device{}()));

        for (i = 0; i < numRams; i++) {
            bloomRams[i]->reset();
        }    
    }

    int getNumRams()
    {
        return numRams;
    }

    long int getBloomBits()
    {
        return bloomRams[0]->getNumBits();
    }

    long int getBloomCapacity()
    {
        return bloomRams[0]->getCapacity();
    }

    float getBloomError()
    {
        return bloomRams[0]->getError();
    }

    int getBloomHashes()
    {
        return bloomRams[0]->getNumHashes();
    }
    

private:
    int entrySize;
    int tupleSize;
    int numRams;
    int * tuplesMapping;
    long int numIntAddr;
    bitarray_t addr;
    vector<BloomFilter*> bloomRams;
};

/*int main() {
    BloomDiscriminator * disc = new BloomDiscriminator(1024, 16, 1000, 0.1);
    
    vector<bool> data = vector<bool>(1024);
    int i;

    for (i = 0; i < 1024; i++) {
        data[i] = i&1;
    }

    disc->train(data);

    cout << "Rank 1 = " << dec << disc->rank(data) << endl;
    disc->info();

    delete disc;
    
    return 0;
}*/