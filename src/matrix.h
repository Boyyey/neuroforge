#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *data;
    int is_view;
} Matrix;

// Creation and destruction
Matrix* matrix_create(size_t rows, size_t cols);
Matrix* matrix_view(Matrix* src, size_t row_start, size_t col_start, size_t rows, size_t cols);
void matrix_free(Matrix* m);

// Basic operations
void matrix_copy(Matrix* dst, const Matrix* src);
void matrix_fill(Matrix* m, float value);
void matrix_random_uniform(Matrix* m, float min, float max);
void matrix_random_normal(Matrix* m, float mean, float stddev);

// Mathematical operations
void matrix_add(Matrix* a, const Matrix* b);
void matrix_subtract(Matrix* a, const Matrix* b);
void matrix_multiply_elementwise(Matrix* a, const Matrix* b);
void matrix_scale(Matrix* m, float scalar);
void matrix_add_scalar(Matrix* m, float scalar);

// BLAS operations
void matrix_multiply(const Matrix* a, const Matrix* b, Matrix* c);
void matrix_transpose(const Matrix* src, Matrix* dst);

// Reduction operations
float matrix_sum(const Matrix* m);
float matrix_max(const Matrix* m);
float matrix_min(const Matrix* m);

// Utility functions
void matrix_print(const Matrix* m, const char* name);
int matrix_equal(const Matrix* a, const Matrix* b, float tolerance);
void matrix_from_array(Matrix* m, const float* data);
void matrix_sqrt(Matrix* m);

// CUDA support
#ifdef USE_CUDA
void matrix_to_gpu(Matrix* m);
void matrix_to_cpu(Matrix* m);
#endif

#endif // MATRIX_H