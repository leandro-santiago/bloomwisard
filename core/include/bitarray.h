/*
 * bitarray.h
 *
 *  Author: Leandro Santiago
 */

#ifndef INCLUDE_BITARRAY_H_
#define INCLUDE_BITARRAY_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct {
	long int num_bits;
	long int bitarray_size;
	uint64_t * bitarray;
} bitarray_t;

//bitarray_t functions
bitarray_t * bitarray_new(long int n_bit);
void bitarray_print(bitarray_t * ba);
void bitarray_free(bitarray_t * ba);

#ifdef __cplusplus
}  // end of extern "C"
#endif

#endif /* INCLUDE_BITARRAY_H_ */