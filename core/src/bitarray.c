/*
* bitarray.c
*
*  Author: Leandro Santiago
*/

#include "bitarray.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bitarray_t * 
bitarray_new(long int n_bit)
{
    bitarray_t * ba = (bitarray_t*) malloc(sizeof(bitarray_t));
	ba->num_bits = n_bit;

	long int n_int = (n_bit >> 6); //divide by 64 bits
	n_int += ((n_bit & 0x3F) > 0); //ceil quotient. If remainder > 0 then sum by 1

	ba->bitarray = (uint64_t *) calloc(n_int, sizeof(uint64_t));
	ba->bitarray_size = n_int;

	return ba;
}

void 
bitarray_print(bitarray_t * ba)
{
    unsigned int i, k, l;
    char * bin_str = (char *) malloc(sizeof(char)*(ba->num_bits+1));
    memset(bin_str, '0', ba->num_bits);

    for (i = 0; i < ba->num_bits; i++) {
        k = i>>6; //divide by 64 bits
        l = 63 - (i & 0x3F); //reminder of 64 bits division

        if (ba->bitarray[k] & (1UL << l)) {
            bin_str[i] = '1';
        }
    }

    bin_str[i] = '\0';

    printf("%s\n", bin_str);

}

void 
bitarray_free(bitarray_t * ba)
{
    if (ba != NULL) {
		free(ba->bitarray);
	}

	free(ba);
}