#ifndef LAYER_H
#define LAYER_H

#include "../matrix.h"

typedef enum {
    LAYER_DENSE,
    LAYER_CONV2D,
    LAYER_RNN,
    LAYER_LSTM,
    LAYER_ATTENTION,
    LAYER_DROPOUT,
    LAYER_BATCHNORM
} LayerType;

typedef struct Layer {
    LayerType type;
    char name[64];
    
    // Parameters
    Matrix* weights;
    Matrix* biases;
    Matrix* running_mean;
    Matrix* running_variance;
    
    // Gradients
    Matrix* grad_weights;
    Matrix* grad_biases;
    
    // State
    Matrix* input;
    Matrix* output;
    Matrix* hidden_state;  // For RNN/LSTM
    
    // Configuration
    float dropout_rate;
    int input_size;
    int output_size;
    int hidden_size;
    int kernel_size;
    int stride;
    int padding;
    int heads;  // For attention
    
    // Activation
    int activation;
    
    // Methods
    void (*forward)(struct Layer* layer, const Matrix* input);
    void (*backward)(struct Layer* layer, const Matrix* output_grad);
    void (*update)(struct Layer* layer, float learning_rate);
    void (*free)(struct Layer* layer);
    
    // Next layer in network
    struct Layer* next;
} Layer;

// Layer creation functions
Layer* dense_layer(int input_size, int output_size, int activation);
Layer* conv2d_layer(int in_channels, int out_channels, int kernel_size, int stride, int padding, int activation);
Layer* rnn_layer(int input_size, int hidden_size, int output_size, int activation);
Layer* attention_layer(int embed_size, int heads);
Layer* dropout_layer(float rate);
Layer* batchnorm_layer(int size);

#endif // LAYER_H