#include "layer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward pass for dense layer
static void dense_forward(Layer* layer, const Matrix* input) {
    // Store input for backward pass
    if (layer->input) matrix_free(layer->input);
    layer->input = matrix_copy(input);
    
    // Allocate output if needed
    if (!layer->output) {
        layer->output = matrix_create(input->rows, layer->output_size);
    }
    
    // Compute output = input * weights + bias
    matrix_multiply(input, layer->weights, layer->output);
    
    // Add bias (broadcasted to each row)
    for (size_t i = 0; i < layer->output->rows; i++) {
        for (size_t j = 0; j < layer->output->cols; j++) {
            layer->output->data[i * layer->output->stride + j] += 
                layer->biases->data[j];
        }
    }
    
    // Apply activation function
    if (layer->activation != ACTIVATION_NONE) {
        activate(layer->output, layer->activation);
    }
}

// Backward pass for dense layer
static void dense_backward(Layer* layer, const Matrix* output_grad) {
    if (!layer->input) return;
    
    // Compute gradient of activation
    Matrix* activation_grad = matrix_copy(output_grad);
    if (layer->activation != ACTIVATION_NONE) {
        activate_derivative(layer->output, activation_grad, layer->activation);
    }
    
    // Compute gradient of weights: input^T * activation_grad
    Matrix* input_t = matrix_create(layer->input->cols, layer->input->rows);
    matrix_transpose(layer->input, input_t);
    
    Matrix* grad_weights = matrix_create(input_t->rows, activation_grad->cols);
    matrix_multiply(input_t, activation_grad, grad_weights);
    
    // Accumulate weight gradients
    matrix_add(layer->grad_weights, grad_weights);
    
    // Compute gradient of biases: sum(activation_grad, axis=0)
    for (size_t i = 0; i < activation_grad->rows; i++) {
        for (size_t j = 0; j < activation_grad->cols; j++) {
            layer->grad_biases->data[j] += 
                activation_grad->data[i * activation_grad->stride + j];
        }
    }
    
    // Compute gradient for previous layer: activation_grad * weights^T
    // (This would be passed to the previous layer in the chain)
    
    // Clean up
    matrix_free(activation_grad);
    matrix_free(input_t);
    matrix_free(grad_weights);
}

// Update parameters for dense layer
static void dense_update(Layer* layer, float learning_rate) {
    // Update weights
    matrix_scale(layer->grad_weights, -learning_rate);
    matrix_add(layer->weights, layer->grad_weights);
    matrix_scale(layer->grad_weights, 0);  // Reset gradients
    
    // Update biases
    matrix_scale(layer->grad_biases, -learning_rate);
    matrix_add(layer->biases, layer->grad_biases);
    matrix_scale(layer->grad_biases, 0);   // Reset gradients
}

// Free dense layer resources
static void dense_free(Layer* layer) {
    if (layer->weights) matrix_free(layer->weights);
    if (layer->biases) matrix_free(layer->biases);
    if (layer->grad_weights) matrix_free(layer->grad_weights);
    if (layer->grad_biases) matrix_free(layer->grad_biases);
    if (layer->input) matrix_free(layer->input);
    if (layer->output) matrix_free(layer->output);
    free(layer);
}

// Create a dense layer
Layer* dense_layer(int input_size, int output_size, int activation) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    memset(layer, 0, sizeof(Layer));
    
    layer->type = LAYER_DENSE;
    strcpy(layer->name, "dense");
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    
    // Initialize weights and biases
    layer->weights = matrix_create(input_size, output_size);
    layer->biases = matrix_create(1, output_size);
    
    // Xavier/Glorot initialization
    float limit = sqrtf(6.0f / (input_size + output_size));
    matrix_random_uniform(layer->weights, -limit, limit);
    matrix_fill(layer->biases, 0.1f);
    
    // Initialize gradients
    layer->grad_weights = matrix_create(input_size, output_size);
    layer->grad_biases = matrix_create(1, output_size);
    matrix_fill(layer->grad_weights, 0.0f);
    matrix_fill(layer->grad_biases, 0.0f);
    
    // Set method pointers
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->update = dense_update;
    layer->free = dense_free;
    
    return layer;
}