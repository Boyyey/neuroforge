#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../matrix.h"

typedef struct {
    char name[64];
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;  // Time step
    
    // Momentum terms
    Matrix** m;  // First moment vector
    Matrix** v;  // Second moment vector
    
    // Parameters and gradients
    Matrix** params;
    Matrix** grads;
    int param_count;
    
    // Methods
    void (*update)(void* optimizer);
    void (*free)(void* optimizer);
} Optimizer;

// Optimizer creation
Optimizer* sgd_optimizer(float learning_rate, float momentum);
Optimizer* adam_optimizer(float learning_rate, float beta1, float beta2, float epsilon);
Optimizer* rmsprop_optimizer(float learning_rate, float decay, float epsilon);

#endif // OPTIMIZER_H