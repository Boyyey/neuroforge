#include "layer.h"
#include "../activations/activation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward pass for dense layer
static void dense_forward(Layer* layer, const Matrix* input) {
    // Store input for backward pass
    if (layer->input) matrix_free(layer->input);
    // Create a copy of the input matrix
    layer->input = matrix_create(input->rows, input->cols);
    matrix_copy(layer->input, input);
    
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
    
    // Store pre-activation values for backward pass
    if (layer->activation != ACTIVATION_NONE) {
        // Store pre-activation values before applying activation
        if (!layer->grad_input) {
            layer->grad_input = matrix_create(layer->output->rows, layer->output->cols);
        }
        matrix_copy(layer->grad_input, layer->output);
        
        // Apply activation function
        activate(layer->output, layer->activation);
    }
}

// Backward pass for dense layer
static void dense_backward(Layer* layer, const Matrix* output_grad) {
    if (!layer->input) return;
    
    // Compute gradient of activation
    Matrix* activation_grad = matrix_create(output_grad->rows, output_grad->cols);
    matrix_copy(activation_grad, output_grad);
    
    if (layer->activation != ACTIVATION_NONE && layer->grad_input) {
        // Use stored pre-activation values for derivative
        activate_derivative(layer->grad_input, activation_grad, layer->activation);
    }
    
    // Compute gradient of weights: input^T * activation_grad
    for (size_t i = 0; i < layer->input->cols; i++) {
        for (size_t j = 0; j < activation_grad->cols; j++) {
            float grad_sum = 0.0f;
            for (size_t k = 0; k < layer->input->rows; k++) {
                grad_sum += layer->input->data[k * layer->input->stride + i] * 
                           activation_grad->data[k * activation_grad->stride + j];
            }
            layer->grad_weights->data[i * layer->grad_weights->stride + j] = grad_sum;
        }
    }
    
    // Compute gradient of biases: sum(activation_grad, axis=0)
    for (size_t i = 0; i < activation_grad->cols; i++) {
        layer->grad_biases->data[i] = 0.0f;
        for (size_t j = 0; j < activation_grad->rows; j++) {
            layer->grad_biases->data[i] += 
                activation_grad->data[j * activation_grad->stride + i];
        }
    }
    
    // Clean up
    matrix_free(activation_grad);
}

// Update parameters for dense layer
static void dense_update(Layer* layer, float learning_rate) {
    // Update weights: weights = weights - learning_rate * grad_weights
    for (size_t i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
        layer->weights->data[i] -= learning_rate * layer->grad_weights->data[i];
    }
    
    // Update biases: biases = biases - learning_rate * grad_biases
    for (size_t i = 0; i < layer->biases->rows * layer->biases->cols; i++) {
        layer->biases->data[i] -= learning_rate * layer->grad_biases->data[i];
    }
    
    // Reset gradients
    matrix_fill(layer->grad_weights, 0.0f);
    matrix_fill(layer->grad_biases, 0.0f);
}

// Free dense layer resources
static void dense_free(Layer* layer) {
    if (layer->weights) matrix_free(layer->weights);
    if (layer->biases) matrix_free(layer->biases);
    if (layer->grad_weights) matrix_free(layer->grad_weights);
    if (layer->grad_biases) matrix_free(layer->grad_biases);
    if (layer->input) matrix_free(layer->input);
    if (layer->output) matrix_free(layer->output);
    if (layer->grad_input) matrix_free(layer->grad_input);
    free(layer);
}

// Create a dense layer
Layer* dense_layer(int input_size, int output_size, ActivationType activation) {
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