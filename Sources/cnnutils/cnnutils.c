#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "cnnutils.h"


/*
             |b00, b01, b02, b03|
             |b10, b11, b12, b13|

|a00, a01|   |c00, c01, c02, c03|
|a10, a11|   |c10, c11, c12, c13|
|a20, a21|   |c20, c21, c22, c23|
|a30, a31|   |c30, c31, c32, c33|
*/

int naive_mmul(
    const float *A,
    const float *B,
    float *C,
    const int a_rows,
    const int b_cols,
    const int a_cols_b_rows
) {
    
    for (int m = 0; m < a_rows; m++) {
        for (int n = 0; n < b_cols; n++) {
            float sum = 0;
            for (int p = 0; p < a_cols_b_rows; p++) {
                sum += A[m * a_cols_b_rows + p] * B[p * b_cols + n];
            }
            C[m * b_cols + n] = sum;
        }
    }
    return 0;
}

void random_set_seed(unsigned int seed) {
    srand(seed);
}

// Uniform distribution in [0..1]
float random_uniform() {
    return (float)((double)rand() / (double)(RAND_MAX));
}

// https://en.wikipedia.org/wiki/Marsaglia_polar_method
float random_normal(float mean, float stdDev) {
    static float spare;
    static bool hasSpare = false;

    if (hasSpare) {
        hasSpare = false;
        return spare * stdDev + mean;
    }

    float u, v, s;
    do {
        u = random_uniform() * 2.0 - 1.0;
        v = random_uniform() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    hasSpare = true;
    return mean + stdDev * u * s;
}
