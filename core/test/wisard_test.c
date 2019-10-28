#include <stdio.h>
#include "wisard.h"

int main(int argc, char ** argv) {
    
    int entry_size = 100;
    int num_tuples = 10;
    
    discriminator_t * disc = discriminator_new(entry_size, num_tuples);
    
    //Create data
    bitarray_t data;
    data.num_bits = 100;
    data.bitarray_size = 2;
    data.bitarray = (uint64_t *) calloc(2, sizeof(uint64_t));

    data.bitarray[0] = 542214;
    data.bitarray[1] = 40000100;

    //Train data
    discriminator_train(disc, &data);

    //discriminator_info(disc);
    
    //Create Test data
    bitarray_t data2;
    data2.num_bits = 100;
    data2.bitarray_size = 2;
    data2.bitarray = (uint64_t *) calloc(2, sizeof(uint64_t));

    data2.bitarray[0] = 542210;
    data2.bitarray[1] = 400682236;

    //Test data
    int r = discriminator_rank(disc, &data2);
    
    printf("Rank = %d\n", r);

    free(data.bitarray);
    free(data2.bitarray);    
    discriminator_free(disc);
    printf("Done!\n");

    return 0;
}