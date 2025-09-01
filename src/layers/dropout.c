#include "layer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Forward pass for dropout layer
static void dropout_forward(Layer* layer, const Matrix* input) {
    if (layer->input) matrix_free(layer->input);
    // Create a copy of the input matrix
    layer->input = matrix_create(input->rows, input->cols);
    matrix_copy(layer->input, input);
    
    if (!layer->output) {
        layer->output = matrix_create(input->rows, input->cols);
    }
    
    if (layer->is_training && layer->dropout_rate > 0) {
        // Create mask and apply dropout during training
        if (!layer->mask) {
            layer->mask = matrix_create(input->rows, input->cols);
        }
        
        float scale = 1.0f / (1.0f - layer->dropout_rate);
        
        for (size_t i = 0; i < input->rows * input->cols; i++) {
            if ((float)rand() / RAND_MAX < layer->dropout_rate) {
                layer->mask->data[i] = 0.0f;
                layer->output->data[i] = 0.0f;
            } else {
                layer->mask->data[i] = scale;
                layer->output->data[i] = input->data[i] * scale;
            }
        }
    } else {
        // During inference, just pass through
        matrix_copy(layer->output, input);
    }
}

// Backward pass for dropout layer
static void dropout_backward(Layer* layer, const Matrix* output_grad) {
    if (!layer->input || !layer->mask) return;
    
    // Allocate grad_input if needed
    if (!layer->grad_input) {
        layer->grad_input = matrix_create(output_grad->rows, output_grad->cols);
    }
    
    // Apply the same mask to gradients
    for (size_t i = 0; i < output_grad->rows * output_grad->cols; i++) {
        layer->grad_input->data[i] = output_grad->data[i] * layer->mask->data[i];
    }
}

// Update parameters for dropout layer (none needed)
static void dropout_update(Layer* layer, float learning_rate) {
    // Dropout has no parameters to update
    (void)layer;        // Suppress unused parameter warning
    (void)learning_rate; // Suppress unused parameter warning
}

// Free dropout layer resources
static void dropout_free(Layer* layer) {
    if (layer->input) matrix_free(layer->input);
    if (layer->output) matrix_free(layer->output);
    if (layer->mask) matrix_free(layer->mask);
    if (layer->grad_input) matrix_free(layer->grad_input);
    free(layer);
}

// Create a dropout layer
Layer* dropout_layer(float rate) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    memset(layer, 0, sizeof(Layer));
    
    layer->type = LAYER_DROPOUT;
    strcpy(layer->name, "dropout");
    layer->dropout_rate = rate;
    layer->is_training = 1;  // Default to training mode
    
    // Set method pointers
    layer->forward = dropout_forward;
    layer->backward = dropout_backward;
    layer->update = dropout_update;
    layer->free = dropout_free;
    
    return layer;
}