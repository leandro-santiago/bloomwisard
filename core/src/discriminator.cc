#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include "bitarray.h"

using namespace std;

class Discriminator {
public:
    Discriminator(int entrySize, int tupleSize) :
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

        //Allocate Ram Memory
        rams = (bitarray_t *) calloc(numRams, sizeof(bitarray_t));
        
        int num_bits = (1UL << tupleSize);//2^tuple_size
        long int bitarray_size = (num_bits >> 6); //divide by 64 bits
        bitarray_size += ((bitarray_size & 0x3F) > 0); //ceil quotient. If remainder > 0 then sum by 1

        for (i = 0; i < numRams; i++) {
            rams[i].num_bits = num_bits;
            rams[i].bitarray_size = bitarray_size;
            rams[i].bitarray = (uint64_t *)calloc(rams[i].bitarray_size, sizeof(uint64_t));
        }
    }

    ~Discriminator() 
    {
        //cout << "Destructor" << endl;
        free(rams);
        free(tuplesMapping);
    }

    void train(bitarray_t * data)
    {
        int i, j, k = 0, i1, i2;
        int addr_pos;
        uint64_t addr;

        for (i = 0; i < numRams; i++) {
            addr_pos = tupleSize-1;
            addr = 0;
            
            for (j = 0; j < tupleSize; j++) {
                i1 = tuplesMapping[k] >> 6;//Divide by 64 to find the bitarray id
                i2 = tuplesMapping[k] & 0x3F;//Obtain remainder to access the bitarray position
            
                addr |= (((data->bitarray[i1] & (1UL << i2)) >> i2) << addr_pos);
                addr_pos--;
                k++;
            }

            i1 = addr >> 6;//Divide by 64 to find the bitarray id
            i2 = addr & 0x3F;//Obtain remainder to access the bitarray position
            rams[i].bitarray[i1] |= (1UL << i2); 
        }
    }

    void train(const vector<bool>& data)
    {
        int i, j, k = 0, i1, i2;
        int addr_pos;
        uint64_t addr;

        for (i = 0; i < numRams; i++) {
            addr_pos = tupleSize-1;
            addr = 0;
            
            for (j = 0; (j < tupleSize) && (k < entrySize); j++) {
                addr |= (data[tuplesMapping[k]] << addr_pos);
                addr_pos--;
                k++;
            }

            i1 = addr >> 6;//Divide by 64 to find the bitarray id
            i2 = addr & 0x3F;//Obtain remainder to access the bitarray position
            rams[i].bitarray[i1] |= (1UL << i2); 
        }
    }

    int rank(bitarray_t * data)
    {
        int rank = 0;
        int i, j, k = 0, i1, i2;
        int addr_pos;
        uint64_t addr;

        for (i = 0; i < numRams; i++) {
            addr_pos = tupleSize-1;
            addr = 0;
            
            for (j = 0; j < tupleSize; j++) {
                i1 = tuplesMapping[k] >> 6;//Divide by 64 to find the bitarray id
                i2 = tuplesMapping[k] & 0x3F;//Obtain remainder to access the bitarray position
            
                addr |= (((data->bitarray[i1] & (1UL << i2)) >> i2) << addr_pos);
                addr_pos--;    
                k++;
            }

            i1 = addr >> 6;//Divide by 64 to find the bitarray id
            i2 = addr & 0x3F;//Obtain remainder to access the bitarray position
            rank += (rams[i].bitarray[i1] & (1UL << i2)) >> i2; 
        }

        return rank;
    }

    int rank(const vector<bool>& data)
    {
        int rank = 0;
        int i, j, k = 0, i1, i2;
        int addr_pos;
        uint64_t addr;

        for (i = 0; i < numRams; i++) {
            addr_pos = tupleSize-1;
            addr = 0;
            
            for (j = 0; (j < tupleSize) && (k < entrySize); j++) {
                addr |= (data[tuplesMapping[k]] << addr_pos);
                addr_pos--;    
                k++;
            }

            i1 = addr >> 6;//Divide by 64 to find the bitarray id
            i2 = addr & 0x3F;//Obtain remainder to access the bitarray position
            rank += (rams[i].bitarray[i1] & (1UL << i2)) >> i2; 
        }

        return rank;
    }

    void info() 
    {
        int i;
        long int totalBits = 0;
        cout << "Entry = " << entrySize << ", Tuples = " << tupleSize << ", RAMs = " << numRams << ", RAM size = " << rams[0].num_bits << " bits" << endl;
        
        for (i = 0; i < numRams; i++) {
           // cout << "RAM " << i << " - " << rams[i].num_bits << " bits" << endl;

            /*for (j = 0; j < rams[i].bitarray_size; j++) {
                cout << rams[i].bitarray[j] << ", ";
            }
            cout << endl;*/

            totalBits += rams[i].num_bits;
        }

        cout << "Total Bits = " << totalBits << endl;
    }

    void reset()
    {
        //Generating pseudo-random mapping
        int i, j;

        for (i = 0; i < entrySize; i++) {
            tuplesMapping[i] = i;
        }
        
        std::shuffle(tuplesMapping, tuplesMapping + entrySize, std::default_random_engine(std::random_device{}()));

        for (i = 0; i < numRams; i++) {
            for (j = 0; j < rams[i].bitarray_size; j++) {
                rams[i].bitarray[j] = 0;
            }
        }    
    }

    long int getRamBits() 
    {
        return rams[0].num_bits;
    }

    int getNumRams()
    {
        return numRams;
    }

private:
    int entrySize;
    int tupleSize;
    int numRams;
    int * tuplesMapping;
    bitarray_t * rams;
};

/*int main(){

    Discriminator * disc = new Discriminator(1024, 16);
    
    vector<bool> data = vector<bool>(1024);
    int i;

    for (i = 0; i < 1024; i++) {
        data[i] = i&1;
    }

    for (i = 0; i < 1024; i++) {
        cout << data[i];
    }    
    cout << endl;

    disc->train(data);

    cout << "Rank=" << dec << disc->rank(data) << endl;
    disc->info();

    delete disc;
    return 0;
}*/