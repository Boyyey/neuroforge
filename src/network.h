#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "layers/layer.h"
#include "optimizers/optimizer.h"

typedef struct {
    Layer* input_layer;
    Layer* output_layer;
    int layer_count;
    
    Optimizer* optimizer;
    float learning_rate;
    
    // Regularization
    float l2_lambda;
    float dropout_rate;
    
    // Training state
    int is_training;
} Network;

// Network creation and management
Network* network_create();
void network_add_layer(Network* net, Layer* layer);
void network_compile(Network* net, Optimizer* optimizer, float l2_lambda);
void network_set_optimizer(Network* net, Optimizer* optimizer);
void network_free(Network* net);

// Forward and backward pass
Matrix* network_forward(Network* net, const Matrix* input);
void network_backward(Network* net, const Matrix* target);
void network_update(Network* net);

// Training functions
float network_train(Network* net, const Matrix* input, const Matrix* target);
float network_test(Network* net, const Matrix* input, const Matrix* target);

// Serialization
void network_save(Network* net, const char* filename);
Network* network_load(const char* filename);

#endif // NETWORK_H