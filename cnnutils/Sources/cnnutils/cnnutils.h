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

#endif /* cnnutils_h */
