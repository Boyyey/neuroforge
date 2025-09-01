#include "layer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward pass for batch normalization layer
static void batchnorm_forward(Layer* layer, const Matrix* input) {
    // Simplified implementation - just pass through for now
    if (layer->input) matrix_free(layer->input);
    layer->input = matrix_create(input->rows, input->cols);
    matrix_copy(layer->input, input);
    
    if (!layer->output) {
        layer->output = matrix_create(input->rows, input->cols);
    }
    matrix_copy(layer->output, input);
}

// Backward pass for batch normalization layer
static void batchnorm_backward(Layer* layer, const Matrix* output_grad) {
    // Simplified implementation
    (void)layer;
    (void)output_grad;
}

// Update parameters for batch normalization layer
static void batchnorm_update(Layer* layer, float learning_rate) {
    // Simplified implementation
    (void)layer;
    (void)learning_rate;
}

// Free batch normalization layer resources
static void batchnorm_free(Layer* layer) {
    if (layer->weights) matrix_free(layer->weights);
    if (layer->biases) matrix_free(layer->biases);
    if (layer->running_mean) matrix_free(layer->running_mean);
    if (layer->running_variance) matrix_free(layer->running_variance);
    if (layer->grad_weights) matrix_free(layer->grad_weights);
    if (layer->grad_biases) matrix_free(layer->grad_biases);
    if (layer->input) matrix_free(layer->input);
    if (layer->output) matrix_free(layer->output);
    free(layer);
}

// Create a batch normalization layer
Layer* batchnorm_layer(int size) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    memset(layer, 0, sizeof(Layer));
    
    layer->type = LAYER_BATCHNORM;
    strcpy(layer->name, "batchnorm");
    layer->input_size = size;
    layer->output_size = size;
    
    // Initialize running statistics
    layer->running_mean = matrix_create(1, size);
    layer->running_variance = matrix_create(1, size);
    matrix_fill(layer->running_mean, 0.0f);
    matrix_fill(layer->running_variance, 1.0f);
    
    // Initialize learnable parameters (gamma and beta)
    layer->weights = matrix_create(1, size);  // gamma (scale)
    layer->biases = matrix_create(1, size);   // beta (shift)
    matrix_fill(layer->weights, 1.0f);
    matrix_fill(layer->biases, 0.0f);
    
    // Initialize gradients
    layer->grad_weights = matrix_create(1, size);
    layer->grad_biases = matrix_create(1, size);
    matrix_fill(layer->grad_weights, 0.0f);
    matrix_fill(layer->grad_biases, 0.0f);
    
    // Set method pointers
    layer->forward = batchnorm_forward;
    layer->backward = batchnorm_backward;
    layer->update = batchnorm_update;
    layer->free = batchnorm_free;
    
    return layer;
}
