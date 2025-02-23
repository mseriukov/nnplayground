#include <math.h>
#include <stdbool.h>
#include "rand.h"

// Linear congruential generator.
uint32_t get_next_and_update_state(random_state_t *state) {
    unsigned int prev = state->prev;
    unsigned int next = (prev * 1103515245L + 12345L) & RAND_MODULUS;
    state->prev = next;
    return next;
}

// Uniform distribution in [0..1]
float random_uniform(random_state_t *state) {
    return (float)((double)get_next_and_update_state(state) / (double)(RAND_MODULUS));
}

// https://en.wikipedia.org/wiki/Marsaglia_polar_method
float random_normal(random_state_t *state, float mean, float stdDev) {
    static float spare;
    static bool hasSpare = false;

    if (hasSpare) {
        hasSpare = false;
        return spare * stdDev + mean;
    }
    float u, v, s;
    do {
        u = random_uniform(state) * 2.0 - 1.0;
        v = random_uniform(state) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    hasSpare = true;
    return mean + stdDev * u * s;
}
