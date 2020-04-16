#include <iostream>
#include <vector>

namespace py = pybind11;

class BloomWisard{
public:
    BloomWisard()
    {
        entrySize = 0;
        tupleSize = 0;
        numDiscriminator = 0;
        bloomDiscriminators = vector<BloomDiscriminator*>();
    }

    BloomWisard(int entrySize, int tupleSize, int numDiscriminator, long int capacity, py::kwargs kwargs) :
    entrySize(entrySize), tupleSize(tupleSize), numDiscriminator(numDiscriminator)
    {
        float error = 0.01;
        int numHash = 0;
        long int bloomSize = 0;
        int i;
        bloomDiscriminators = vector<BloomDiscriminator*>(numDiscriminator);

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

        for (i = 0; i < numDiscriminator; i++) {
            bloomDiscriminators[i] = new BloomDiscriminator(entrySize, tupleSize, capacity, error, numHash, bloomSize);
        }
    }

    ~BloomWisard()
    {
        unsigned int i;
        for (i = 0; i < bloomDiscriminators.size(); i++) {
            delete bloomDiscriminators[i];
        }
    }

    void train(const vector<vector<bool>>& data, const vector<int>& label)
    {
        unsigned int i;

        for (i = 0; i < label.size(); i++) {
            bloomDiscriminators[label[i]]->train(data[i]);
        }
    }

    int rank(const vector<bool>& data)
    {
        int i;
        int label = 0;   
        int max_resp = 0;
        int resp;

        for (i = 0; i < numDiscriminator; i++) {
            resp = bloomDiscriminators[i]->rank(data);

            if (resp > max_resp) {
                max_resp = resp;
                label = i;
            }
        }

        return label;
    }

    py::array_t<int> rank(const vector<vector<bool>>& data)
    {
        //Initialize results with numpy array
        /*py::array_t<int> a({ data.size(), data.size() });
        auto result = a.mutable_unchecked();*/
        py::array_t<int> a({data.size()});
        auto label = a.mutable_unchecked();

        unsigned int i;
        int j, max_resp, resp;
        
        for (i = 0; i < data.size(); i++) {
            max_resp = 0;
            for ( j = 0; j < numDiscriminator; j++) {
                resp = bloomDiscriminators[j]->rank(data[i]);

                if (resp > max_resp) {
                    max_resp = resp;
                    label(i) = j;
                }
            }
        }

        return a;
    }

    void info()
    {
        cout << "Number of Bloom Discriminators = " << numDiscriminator << endl;
        int i;

        for (i = 0; i < numDiscriminator; i++) {
            cout << "Bloom Discriminator " << i << ": ";
            bloomDiscriminators[i]->info();
        }
    }

    py::array_t<unsigned long int> stats()
    {
        py::array_t<unsigned long int> a({5});
        auto stats = a.mutable_unchecked();

        int numRams = bloomDiscriminators[0]->getNumRams();
        long int bloomSize = bloomDiscriminators[0]->getBloomBits();
        long int totalRamBits = numRams * bloomSize; 
        long int totalBits = numDiscriminator * totalRamBits;

        stats(0) = numRams;
        stats(1) = bloomSize;
        stats(2) = totalRamBits;
        stats(3) = totalBits;
        stats(4) = bloomDiscriminators[0]->getBloomHashes();

        return a;
    }

    float getError()
    {
        return bloomDiscriminators[0]->getBloomError();
    }

    void reset()
    {
        int i;

        for (i = 0; i < numDiscriminator; i++) {
            bloomDiscriminators[i]->reset();
        }
    }

private:
    int entrySize;
    int tupleSize;
    int numDiscriminator;
    vector<BloomDiscriminator*> bloomDiscriminators;
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