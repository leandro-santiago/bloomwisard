#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <unordered_map>

using namespace std;

class DictDiscriminator {
public:
    DictDiscriminator(int entrySize, int tupleSize) :
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
        rams = vector<unordered_map<unsigned int, bool>>(numRams);
    }

    ~DictDiscriminator() 
    {
        //cout << "Destructor" << endl;
        free(tuplesMapping);
    }

    void train(const vector<bool>& data)
    {
        int i, j, k = 0;
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

            auto it = rams[i].find(addr);
            if (it == rams[i].end()) {
                rams[i].insert(pair<unsigned int, bool>(addr, 1));
            }             
        }
    }

    int rank(const vector<bool>& data)
    {
        int rank = 0;
        int i, j, k = 0;
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

            auto it = rams[i].find(addr);
            if (it != rams[i].end()) {
                rank += it->second;
            }
        }

        return rank;
    }

    void info() 
    {
        int i;
        unsigned int totalPos = 0;
        cout << "Entry = " << entrySize << ", Tuples = " << tupleSize << ", RAMs = " << numRams << endl;
        
        for (i = 0; i < numRams; i++) {
           // cout << "RAM " << i << " - " << rams[i].num_bits << " bits" << endl;

            /*for (j = 0; j < rams[i].bitarray_size; j++) {
                cout << rams[i].bitarray[j] << ", ";
            }
            cout << endl;*/

            totalPos += rams[i].size();
        }

        cout << "Total Positions = " << totalPos <<  endl;
    }

    long int getRamPos() 
    {
        unsigned int i;
        long int totalPos = 0;

        for (i = 0; i < rams.size(); i++) {
            totalPos += rams[i].size();
        }

        return totalPos;
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
    vector<unordered_map<unsigned int, bool>> rams;
};
