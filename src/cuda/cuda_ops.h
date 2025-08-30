#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include "../matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

// Check CUDA availability
int cuda_available();

// Memory management
void cuda_matrix_alloc(Matrix* m);
void cuda_matrix_free(Matrix* m);
void cuda_matrix_copy_to_gpu(const Matrix* host_src, Matrix* device_dst);
void cuda_matrix_copy_to_cpu(const Matrix* device_src, Matrix* host_dst);

// CUDA-accelerated operations
void cuda_matrix_add(Matrix* a, const Matrix* b);
void cuda_matrix_multiply(const Matrix* a, const Matrix* b, Matrix* c);
void cuda_matrix_conv2d(const Matrix* input, const Matrix* kernels, 
                        Matrix* output, int stride, int padding);
void cuda_matrix_max_pool(const Matrix* input, Matrix* output, 
                         int pool_size, int stride);
void cuda_matrix_softmax(Matrix* m);
void cuda_matrix_relu(Matrix* m);
void cuda_matrix_sigmoid(Matrix* m);

#ifdef __cplusplus
}
#endif

#endif // CUDA_OPS_H