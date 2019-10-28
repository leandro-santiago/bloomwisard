#include <iostream>
#include <vector>

using namespace std;

class DictWisard {
public:
    DictWisard() {
        entrySize = 0;
        tupleSize = 0;
        numDiscriminator = 0;
        discriminators = vector<DictDiscriminator*>();
    }

    DictWisard(int entrySize, int tupleSize, int numDiscriminator) :
    entrySize(entrySize), tupleSize(tupleSize), numDiscriminator(numDiscriminator)
    {
        int i;
        discriminators = vector<DictDiscriminator*>(numDiscriminator);

        for (i = 0; i < numDiscriminator; i++) {
            discriminators[i] = new DictDiscriminator(entrySize, tupleSize);
        }
    }

    ~DictWisard()
    {
        unsigned int i;
        for (i = 0; i < discriminators.size(); i++) {
            delete discriminators[i];
        }
    }

    void addDiscriminator()
    {
        discriminators.emplace_back(new DictDiscriminator(entrySize, tupleSize));
        numDiscriminator++;
    }

    void train(const vector<vector<bool>>& data, const vector<int>& label)
    {
        unsigned int i;

        for (i = 0; i < label.size(); i++) {
            discriminators[label[i]]->train(data[i]);
        }
    }

    int rank(const vector<bool>& data)
    {
        int i;
        int label = 0;   
        int max_resp = 0;
        int resp;

        for (i = 0; i < numDiscriminator; i++) {
            resp = discriminators[i]->rank(data);

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
                resp = discriminators[j]->rank(data[i]);

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
        cout << "Number of Discriminators = " << numDiscriminator << endl;
        int i;

        for (i = 0; i < numDiscriminator; i++) {
            cout << "Discriminator " << i << ": ";
            discriminators[i]->info();
        }
    }

    py::array_t<unsigned long int> stats()
    {
        py::array_t<unsigned long int> a({3});
        auto stats = a.mutable_unchecked();

        int numRams = discriminators[0]->getNumRams();
        int i;
        long int totalPos = 0;

        for (i = 0; i < numDiscriminator; i++) {
            totalPos += discriminators[0]->getRamPos();
        }

        stats(0) = numRams;
        stats(1) = totalPos;
        stats(2) = (totalPos * (sizeof(unsigned int) + sizeof(bool))) << 3;
       
        return a;
    }

private:
    int entrySize;
    int tupleSize;
    int numDiscriminator;
    vector<DictDiscriminator*> discriminators;
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