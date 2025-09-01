#include "optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Adam update function
static void adam_update(void* optimizer_ptr) {
    Optimizer* optimizer = (Optimizer*)optimizer_ptr;
    
    optimizer->t++;
    
    float beta1_t = powf(optimizer->beta1, optimizer->t);
    float beta2_t = powf(optimizer->beta2, optimizer->t);
    float lr_t = optimizer->learning_rate * sqrtf(1 - beta2_t) / (1 - beta1_t);
    
    for (int i = 0; i < optimizer->param_count; i++) {
        Matrix* param = optimizer->params[i];
        Matrix* grad = optimizer->grads[i];
        Matrix* m = optimizer->m[i];
        Matrix* v = optimizer->v[i];
        
        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
        matrix_scale(m, optimizer->beta1);
        matrix_scale(grad, 1.0f - optimizer->beta1);
        matrix_add(m, grad);
        matrix_scale(grad, 1.0f / (1.0f - optimizer->beta1));  // Restore grad
        
        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
        Matrix* grad_squared = matrix_create(grad->rows, grad->cols);
        matrix_copy(grad_squared, grad);
        matrix_multiply_elementwise(grad_squared, grad_squared);
        
        matrix_scale(v, optimizer->beta2);
        matrix_scale(grad_squared, 1.0f - optimizer->beta2);
        matrix_add(v, grad_squared);
        matrix_free(grad_squared);
        
        // Compute bias-corrected moments
        Matrix* m_hat = matrix_create(m->rows, m->cols);
        matrix_copy(m_hat, m);
        matrix_scale(m_hat, 1.0f / (1.0f - beta1_t));
        
        Matrix* v_hat = matrix_create(v->rows, v->cols);
        matrix_copy(v_hat, v);
        matrix_scale(v_hat, 1.0f / (1.0f - beta2_t));
        
        // Update parameters: param = param - lr_t * m_hat / (sqrt(v_hat) + epsilon)
        for (size_t j = 0; j < param->rows * param->cols; j++) {
            float denominator = sqrtf(v_hat->data[j]) + optimizer->epsilon;
            param->data[j] -= lr_t * m_hat->data[j] / denominator;
        }
        
        matrix_free(m_hat);
        matrix_free(v_hat);
        matrix_scale(grad, 0.0f);  // Reset gradients
    }
}

// Free Adam optimizer
static void adam_free(void* optimizer_ptr) {
    Optimizer* optimizer = (Optimizer*)optimizer_ptr;
    
    // Free moment vectors
    for (int i = 0; i < optimizer->param_count; i++) {
        if (optimizer->m[i]) matrix_free(optimizer->m[i]);
        if (optimizer->v[i]) matrix_free(optimizer->v[i]);
    }
    
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer);
}

// Create Adam optimizer
Optimizer* adam_optimizer(float learning_rate, float beta1, float beta2, float epsilon) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    memset(optimizer, 0, sizeof(Optimizer));
    
    strcpy(optimizer->name, "adam");
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->t = 0;
    
    optimizer->update = adam_update;
    optimizer->free = adam_free;
    
    return optimizer;
}