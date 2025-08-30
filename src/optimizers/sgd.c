#include "optimizer.h"
#include <stdlib.h>
#include <string.h>

// SGD update function
static void sgd_update(void* optimizer_ptr) {
    Optimizer* optimizer = (Optimizer*)optimizer_ptr;
    
    for (int i = 0; i < optimizer->param_count; i++) {
        // param = param - learning_rate * grad
        matrix_scale(optimizer->grads[i], -optimizer->learning_rate);
        matrix_add(optimizer->params[i], optimizer->grads[i]);
        matrix_scale(optimizer->grads[i], 0.0f);  // Reset gradients
    }
    
    optimizer->t++;
}

// Free SGD optimizer
static void sgd_free(void* optimizer_ptr) {
    Optimizer* optimizer = (Optimizer*)optimizer_ptr;
    
    // Note: We don't own the params and grads, just references
    free(optimizer);
}

// Create SGD optimizer
Optimizer* sgd_optimizer(float learning_rate, float momentum) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    memset(optimizer, 0, sizeof(Optimizer));
    
    strcpy(optimizer->name, "sgd");
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = momentum;  // Using beta1 for momentum
    optimizer->epsilon = 1e-8f;
    optimizer->t = 0;
    
    optimizer->update = sgd_update;
    optimizer->free = sgd_free;
    
    return optimizer;
}