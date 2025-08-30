#include "layer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward pass for attention layer
static void attention_forward(Layer* layer, const Matrix* input) {
    // Simplified self-attention implementation
    // input shape: [batch_size, seq_len, embed_size]
    
    if (layer->input) matrix_free(layer->input);
    layer->input = matrix_copy(input);
    
    // For simplicity, we'll assume input is already projected to Q, K, V
    // In a real implementation, we would have learnable projection matrices
    
    // Compute attention scores: Q * K^T / sqrt(d_k)
    Matrix* scores = matrix_create(input->rows, input->rows); // [seq_len, seq_len]
    
    // Simplified: just use input as Q, K, V
    Matrix* k_t = matrix_create(input->cols, input->rows);
    matrix_transpose(input, k_t);
    
    matrix_multiply(input, k_t, scores);
    matrix_scale(scores, 1.0f / sqrtf(input->cols));
    
    // Apply softmax to get attention weights
    activate(scores, ACTIVATION_SOFTMAX);
    
    // Apply attention to values: weights * V
    if (!layer->output) {
        layer->output = matrix_create(input->rows, input->cols);
    }
    
    matrix_multiply(scores, input, layer->output);
    
    // Clean up
    matrix_free(scores);
    matrix_free(k_t);
}

// Backward pass for attention layer
static void attention_backward(Layer* layer, const Matrix* output_grad) {
    // Implementation would go here
    // This is a complex operation that requires careful implementation
}

// Update parameters for attention layer
static void attention_update(Layer* layer, float learning_rate) {
    // Implementation would go here
}

// Free attention layer resources
static void attention_free(Layer* layer) {
    if (layer->weights) matrix_free(layer->weights);
    if (layer->biases) matrix_free(layer->biases);
    if (layer->grad_weights) matrix_free(layer->grad_weights);
    if (layer->grad_biases) matrix_free(layer->grad_biases);
    if (layer->input) matrix_free(layer->input);
    if (layer->output) matrix_free(layer->output);
    free(layer);
}

// Create an attention layer
Layer* attention_layer(int embed_size, int heads) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    memset(layer, 0, sizeof(Layer));
    
    layer->type = LAYER_ATTENTION;
    strcpy(layer->name, "attention");
    layer->input_size = embed_size;
    layer->output_size = embed_size;
    layer->heads = heads;
    
    // Initialize projection matrices for Q, K, V
    // For simplicity, we'll just use one set of weights for now
    int proj_size = embed_size * 3;  // Q, K, V concatenated
    layer->weights = matrix_create(embed_size, proj_size);
    
    // Xavier initialization
    float limit = sqrtf(6.0f / (embed_size + proj_size));
    matrix_random_uniform(layer->weights, -limit, limit);
    
    // Initialize gradients
    layer->grad_weights = matrix_create(embed_size, proj_size);
    matrix_fill(layer->grad_weights, 0.0f);
    
    // Set method pointers
    layer->forward = attention_forward;
    layer->backward = attention_backward;
    layer->update = attention_update;
    layer->free = attention_free;
    
    return layer;
}