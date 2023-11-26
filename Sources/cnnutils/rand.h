#ifndef rand_h
#define rand_h

#include <stdint.h>

#define RAND_MODULUS 0x7FFFFFFF

struct random_state {
    uint32_t prev;
};
typedef struct random_state random_state_t;

float random_uniform(random_state_t *state);
float random_normal(random_state_t *state, float mean, float stdDev);

#endif /* rand_h */
