#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>

// Check if CUDA is available
int cuda_available() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// CUDA error checking macro
#define CHECK_CUDA(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while (0)

// CUDA kernel for element-wise addition
__global__ void cuda_matrix_add_kernel(float* a, const float* b, size_t rows, size_t cols, size_t stride_a, size_t stride_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows && j < cols) {
        a[i * stride_a + j] += b[i * stride_b + j];
    }
}

// CUDA kernel for ReLU activation
__global__ void cuda_matrix_relu_kernel(float* m, size_t rows, size_t cols, size_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows && j < cols) {
        float val = m[i * stride + j];
        m[i * stride + j] = val > 0 ? val : 0;
    }
}

// CUDA kernel for sigmoid activation
__global__ void cuda_matrix_sigmoid_kernel(float* m, size_t rows, size_t cols, size_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows && j < cols) {
        m[i * stride + j] = 1.0f / (1.0f + expf(-m[i * stride + j]));
    }
}

// CUDA kernel for softmax activation (per row)
__global__ void cuda_matrix_softmax_kernel(float* m, size_t rows, size_t cols, size_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < rows) {
        // Find max value in row for numerical stability
        float max_val = m[i * stride];
        for (int j = 1; j < cols; j++) {
            if (m[i * stride + j] > max_val) {
                max_val = m[i * stride + j];
            }
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = expf(m[i * stride + j] - max_val);
            m[i * stride + j] = val;
            sum += val;
        }
        
        // Normalize
        for (int j = 0; j < cols; j++) {
            m[i * stride + j] /= sum;
        }
    }
}

// Allocate matrix on GPU
void cuda_matrix_alloc(Matrix* m) {
    CHECK_CUDA(cudaMalloc(&m->data, m->rows * m->cols * sizeof(float)));
}

// Free matrix from GPU
void cuda_matrix_free(Matrix* m) {
    CHECK_CUDA(cudaFree(m->data));
}

// Copy matrix from host to device
void cuda_matrix_copy_to_gpu(const Matrix* host_src, Matrix* device_dst) {
    CHECK_CUDA(cudaMemcpy(device_dst->data, host_src->data, 
                         host_src->rows * host_src->cols * sizeof(float),
                         cudaMemcpyHostToDevice));
}

// Copy matrix from device to host
void cuda_matrix_copy_to_cpu(const Matrix* device_src, Matrix* host_dst) {
    CHECK_CUDA(cudaMemcpy(host_dst->data, device_src->data,
                         device_src->rows * device_src->cols * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

// Matrix addition on GPU
void cuda_matrix_add(Matrix* a, const Matrix* b) {
    dim3 blockSize(16, 16);
    dim3 gridSize((a->rows + blockSize.x - 1) / blockSize.x,
                  (a->cols + blockSize.y - 1) / blockSize.y);
    
    cuda_matrix_add_kernel<<<gridSize, blockSize>>>(
        a->data, b->data, a->rows, a->cols, a->stride, b->stride
    );
    CHECK_CUDA(cudaGetLastError());
}

// Matrix multiplication using cuBLAS
void cuda_matrix_multiply(const Matrix* a, const Matrix* b, Matrix* c) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                c->cols, c->rows, a->cols,
                &alpha,
                b->data, b->stride,
                a->data, a->stride,
                &beta,
                c->data, c->stride);
    
    cublasDestroy(handle);
}

// ReLU activation on GPU
void cuda_matrix_relu(Matrix* m) {
    dim3 blockSize(16, 16);
    dim3 gridSize((m->rows + blockSize.x - 1) / blockSize.x,
                  (m->cols + blockSize.y - 1) / blockSize.y);
    
    cuda_matrix_relu_kernel<<<gridSize, blockSize>>>(
        m->data, m->rows, m->cols, m->stride
    );
    CHECK_CUDA(cudaGetLastError());
}

// Sigmoid activation on GPU
void cuda_matrix_sigmoid(Matrix* m) {
    dim3 blockSize(16, 16);
    dim3 gridSize((m->rows + blockSize.x - 1) / blockSize.x,
                  (m->cols + blockSize.y - 1) / blockSize.y);
    
    cuda_matrix_sigmoid_kernel<<<gridSize, blockSize>>>(
        m->data, m->rows, m->cols, m->stride
    );
    CHECK_CUDA(cudaGetLastError());
}

// Softmax activation on GPU
void cuda_matrix_softmax(Matrix* m) {
    dim3 blockSize(256);
    dim3 gridSize((m->rows + blockSize.x - 1) / blockSize.x);
    
    cuda_matrix_softmax_kernel<<<gridSize, blockSize>>>(
        m->data, m->rows, m->cols, m->stride
    );
    CHECK_CUDA(cudaGetLastError());
}