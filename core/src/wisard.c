/*
 * wisard.cc
 *
 *  Author: Leandro Santiago
 */

#include <time.h>
#include "wisard.h"
#include <stdio.h>

void
random_shuffle(int * array, int n)
{
	srand(time(NULL));
	int i, j, temp;

	for (i = n - 1; i > 0; i--) {
		j = rand()%(i+1);

		temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
}

discriminator_t * 
discriminator_new(int entry_size, int tuple_size)
{
    discriminator_t * disc = (discriminator_t*) malloc(sizeof(discriminator_t));
	disc->entry_size = entry_size;
    disc->tuple_size = tuple_size;
    disc->num_rams = entry_size/tuple_size + ((entry_size%tuple_size) > 0);
    disc->tuples_mapping = (int *)calloc(entry_size, sizeof(int));
    
    //Generating pseudo-random mapping
    int i;

    for (i = 0; i < entry_size; i++) {
        disc->tuples_mapping[i] = i;
    }
    random_shuffle(disc->tuples_mapping, entry_size);
    
    printf("%d\n", entry_size);
    for (i = 0; i < entry_size; i++) {
        printf("%d,", disc->tuples_mapping[i]);
    }

    printf("\n");

    //Allocate Ram Memory
    disc->rams = (bitarray_t *) calloc(disc->num_rams, sizeof(bitarray_t));
    
    int num_bits = (1UL << tuple_size);//2^tuple_size
    long int bitarray_size = (num_bits >> 6); //divide by 64 bits
	bitarray_size += ((bitarray_size & 0x3F) > 0); //ceil quotient. If remainder > 0 then sum by 1

    for (i = 0; i < disc->num_rams; i++) {
        disc->rams[i].num_bits = num_bits;
        disc->rams[i].bitarray_size = bitarray_size;
        disc->rams[i].bitarray = (uint64_t *)calloc(disc->rams[i].bitarray_size, sizeof(uint64_t));
    }

	return disc;
}

void 
discriminator_info(discriminator_t * disc)
{
    int i, j;

    printf("Entry = %d, Tuples = %d, RAMs = %d\n", disc->entry_size, disc->tuple_size, disc->num_rams);

    for (i = 0; i < disc->num_rams; i++) {
        printf("RAM %d - %ld bits\n", i, disc->rams[i].num_bits);

        for (j = 0; j < disc->rams[i].bitarray_size; j++) {
            printf("%lu, ", disc->rams[i].bitarray[j]);
        }
        printf("\n");
    }
}

void 
discriminator_train(discriminator_t * disc, bitarray_t * data)
{
    int i, j, k = 0, i1, i2;
    int addr_pos;
    uint64_t addr;

    for (i = 0; i < disc->num_rams; i++) {
        addr_pos = disc->tuple_size-1;
        addr = 0;
        
        for (j = 0; j < disc->tuple_size; j++) {
            i1 = disc->tuples_mapping[k] >> 6;//Divide by 64 to find the bitarray id
            i2 = disc->tuples_mapping[k] & 0x3F;//Obtain remainder to access the bitarray position
        
            addr |= (((data->bitarray[i1] & (1UL << i2)) >> i2) << addr_pos);
            addr_pos--;
            k++;
        }

        i1 = addr >> 6;//Divide by 64 to find the bitarray id
        i2 = addr & 0x3F;//Obtain remainder to access the bitarray position
        disc->rams[i].bitarray[i1] |= (1UL << i2); 
    }
}

int 
discriminator_rank(discriminator_t * disc, bitarray_t * data)
{
    int rank = 0;
    int i, j, k = 0, i1, i2;
    int addr_pos;
    uint64_t addr;

    for (i = 0; i < disc->num_rams; i++) {
        addr_pos = disc->tuple_size-1;
        addr = 0;
        
        for (j = 0; j < disc->tuple_size; j++) {
            i1 = disc->tuples_mapping[k] >> 6;//Divide by 64 to find the bitarray id
            i2 = disc->tuples_mapping[k] & 0x3F;//Obtain remainder to access the bitarray position
        
            addr |= (((data->bitarray[i1] & (1UL << i2)) >> i2) << addr_pos);
            addr_pos--;    
            k++;
        }

        i1 = addr >> 6;//Divide by 64 to find the bitarray id
        i2 = addr & 0x3F;//Obtain remainder to access the bitarray position
        rank += (disc->rams[i].bitarray[i1] & (1UL << i2)) >> i2; 
    }

    return rank;
}

void 
discriminator_free(discriminator_t * disc)
{   
    free(disc->rams);
    free(disc->tuples_mapping);
    free(disc);
}
