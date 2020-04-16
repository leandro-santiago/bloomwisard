/*
 * wisard.h
 *
 *  Author: Leandro Santiago
 */

#ifndef INCLUDE_WISARD_H_
#define INCLUDE_WISARD_H_

#include <stdlib.h>
#include "bitarray.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int entry_size;
    int tuple_size;
    int num_rams;
    int * tuples_mapping;
    bitarray_t * rams;
} discriminator_t;

//discriminator_t functions
discriminator_t * discriminator_new(int entry_size, int tuple_size);
void discriminator_info(discriminator_t * disc);
void discriminator_train(discriminator_t * disc, bitarray_t * data);
int discriminator_rank(discriminator_t * disc, bitarray_t * data);
void discriminator_free(discriminator_t * disc);

#ifdef __cplusplus
}  // end of extern "C"
#endif

#endif /* INCLUDE_WISARD_H_ */
