#ifndef cnnutils_h
#define cnnutils_h

int naive_mmul(
    const float *A,
    const float *B,
    float *C,
    const int a_rows,
    const int b_cols,
    const int a_cols_b_rows
);

void random_set_seed(unsigned int seed);
float random_uniform();
float random_normal(float mean, float stdDev);


#endif /* cnnutils_h */
