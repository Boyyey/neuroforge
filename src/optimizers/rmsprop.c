#include "optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// RMSProp update function
static void rmsprop_update(void* optimizer_ptr) {
    Optimizer* optimizer = (Optimizer*)optimizer_ptr;
    
    for (int i = 0; i < optimizer->param_count; i++) {
        Matrix* param = optimizer->params[i];
        Matrix* grad = optimizer->grads[i];
        Matrix* cache = optimizer->v[i];  // Using v for cache in RMSProp
        
        // Update cache: cache = decay * cache + (1 - decay) * grad^2
        matrix_multiply_elementwise(grad, grad);  // grad = grad^2
        matrix_scale(cache, optimizer->beta1);     // decay is stored in beta1
        matrix_scale(grad, 1.0f - optimizer->beta1);
        matrix_add(cache, grad);
        matrix_sqrt(grad);  // Restore grad to original values: grad = sqrt(grad^2)
        matrix_scale(grad, 1.0f / (1.0f - optimizer->beta1));  // Complete restoration
        
        // Update parameters: param = param - learning_rate * grad / (sqrt(cache) + epsilon)
        for (size_t j = 0; j < param->rows * param->cols; j++) {
            float denominator = sqrtf(cache->data[j]) + optimizer->epsilon;
            param->data[j] -= optimizer->learning_rate * grad->data[j] / denominator;
        }
        
        // Reset gradients
        matrix_scale(grad, 0.0f);
    }
    
    optimizer->t++;
}

// Free RMSProp optimizer
static void rmsprop_free(void* optimizer_ptr) {
    Optimizer* optimizer = (Optimizer*)optimizer_ptr;
    
    // Free cache vectors
    for (int i = 0; i < optimizer->param_count; i++) {
        if (optimizer->v[i]) matrix_free(optimizer->v[i]);
    }
    
    free(optimizer->v);
    free(optimizer);
}

// Create RMSProp optimizer
Optimizer* rmsprop_optimizer(float learning_rate, float decay, float epsilon) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    memset(optimizer, 0, sizeof(Optimizer));
    
    strcpy(optimizer->name, "rmsprop");
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = decay;    // Using beta1 for decay rate
    optimizer->epsilon = epsilon;
    optimizer->t = 0;
    
    optimizer->update = rmsprop_update;
    optimizer->free = rmsprop_free;
    
    return optimizer;
}