#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

Matrix* matrix_create(size_t rows, size_t cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->stride = cols;
    m->is_view = 0;
    
    #ifdef USE_CUDA
    if (cuda_available()) {
        cuda_matrix_alloc(m);
    } else {
        m->data = (float*)calloc(rows * cols, sizeof(float));
    }
    #else
    m->data = (float*)calloc(rows * cols, sizeof(float));
    #endif
    
    return m;
}

Matrix* matrix_view(Matrix* src, size_t row_start, size_t col_start, 
                   size_t rows, size_t cols) {
    assert(row_start + rows <= src->rows);
    assert(col_start + cols <= src->cols);
    
    Matrix* view = (Matrix*)malloc(sizeof(Matrix));
    view->rows = rows;
    view->cols = cols;
    view->stride = src->stride;  // View must use parent's stride for correct indexing
    view->is_view = 1;
    view->data = src->data + row_start * src->stride + col_start;
    
    return view;
}

void matrix_free(Matrix* m) {
    if (!m) return;
    
    if (!m->is_view) {
        #ifdef USE_CUDA
        if (cuda_available()) {
            cuda_matrix_free(m);
        } else {
            free(m->data);
        }
        #else
        free(m->data);
        #endif
    }
    
    free(m);
}

void matrix_copy(Matrix* dst, const Matrix* src) {
    assert(dst->rows == src->rows);
    assert(dst->cols == src->cols);
    
    if (dst->rows == src->rows && dst->cols == src->cols) {
        for (size_t i = 0; i < src->rows; i++) {
            for (size_t j = 0; j < src->cols; j++) {
                dst->data[i * dst->stride + j] = src->data[i * src->stride + j];
            }
        }
    }
}

void matrix_fill(Matrix* m, float value) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] = value;
        }
    }
}

void matrix_random_uniform(Matrix* m, float min, float max) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] = min + (max - min) * ((float)rand() / RAND_MAX);
        }
    }
}

void matrix_random_normal(Matrix* m, float mean, float stddev) {
    // Box-Muller transform for normal distribution
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            m->data[i * m->stride + j] = mean + stddev * z;
        }
    }
}

void matrix_add(Matrix* a, const Matrix* b) {
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    
    #ifdef USE_CUDA
    if (cuda_available()) {
        cuda_matrix_add(a, b);
        return;
    }
    #endif
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            a->data[i * a->stride + j] += b->data[i * b->stride + j];
        }
    }
}

void matrix_subtract(Matrix* a, const Matrix* b) {
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            a->data[i * a->stride + j] -= b->data[i * b->stride + j];
        }
    }
}

void matrix_multiply_elementwise(Matrix* a, const Matrix* b) {
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            a->data[i * a->stride + j] *= b->data[i * b->stride + j];
        }
    }
}

void matrix_scale(Matrix* m, float scalar) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] *= scalar;
        }
    }
}

void matrix_add_scalar(Matrix* m, float scalar) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] += scalar;
        }
    }
}

void matrix_multiply(const Matrix* a, const Matrix* b, Matrix* c) {
    assert(a->cols == b->rows);
    assert(a->rows == c->rows);
    assert(b->cols == c->cols);
    
    #ifdef USE_CUDA
    if (cuda_available()) {
        cuda_matrix_multiply(a, b, c);
        return;
    }
    #endif
    
    // Use simple triple loop for now - could be optimized with BLAS
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->cols; k++) {
                sum += a->data[i * a->stride + k] * b->data[k * b->stride + j];
            }
            c->data[i * c->stride + j] = sum;
        }
    }
}

void matrix_transpose(const Matrix* src, Matrix* dst) {
    assert(src->rows == dst->cols);
    assert(src->cols == dst->rows);
    
    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            dst->data[j * dst->stride + i] = src->data[i * src->stride + j];
        }
    }
}

float matrix_sum(const Matrix* m) {
    float sum = 0.0f;
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            sum += m->data[i * m->stride + j];
        }
    }
    return sum;
}

float matrix_max(const Matrix* m) {
    float max_val = m->data[0];
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            if (m->data[i * m->stride + j] > max_val) {
                max_val = m->data[i * m->stride + j];
            }
        }
    }
    return max_val;
}

float matrix_min(const Matrix* m) {
    float min_val = m->data[0];
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            if (m->data[i * m->stride + j] < min_val) {
                min_val = m->data[i * m->stride + j];
            }
        }
    }
    return min_val;
}

void matrix_print(const Matrix* m, const char* name) {
    printf("%s (%zux%zu):\n", name, m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            printf("%8.4f ", m->data[i * m->stride + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int matrix_equal(const Matrix* a, const Matrix* b, float tolerance) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return 0;
    }
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            if (fabs(a->data[i * a->stride + j] - b->data[i * b->stride + j]) > tolerance) {
                return 0;
            }
        }
    }
    
    return 1;
}

void matrix_from_array(Matrix* m, const float* data) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] = data[i * m->cols + j];
        }
    }
}

void matrix_sqrt(Matrix* m) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] = sqrtf(m->data[i * m->stride + j]);
        }
    }
}