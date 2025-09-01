#include "network.h"
#include "layers/layer.h"
#include "activations/activation.h"
#include <stdlib.h>
#include <string.h>

Network* network_create() {
    Network* net = (Network*)malloc(sizeof(Network));
    memset(net, 0, sizeof(Network));
    return net;
}

void network_add_layer(Network* net, Layer* layer) {
    if (!net->input_layer) {
        net->input_layer = layer;
        net->output_layer = layer;
    } else {
        net->output_layer->next = layer;
        net->output_layer = layer;
    }
    net->layer_count++;
}

void network_compile(Network* net, Optimizer* optimizer, float l2_lambda) {
    net->optimizer = optimizer;
    net->l2_lambda = l2_lambda;
    
    // Initialize optimizer with network parameters
    if (net->optimizer) {
        // Count total parameters
        int total_params = 0;
        Layer* layer = net->input_layer;
        while (layer) {
            if (layer->weights) total_params++;
            if (layer->biases) total_params++;
            layer = layer->next;
        }
        
        // Allocate parameter arrays
        net->optimizer->params = (Matrix**)malloc(total_params * sizeof(Matrix*));
        net->optimizer->grads = (Matrix**)malloc(total_params * sizeof(Matrix*));
        net->optimizer->m = (Matrix**)malloc(total_params * sizeof(Matrix*));
        net->optimizer->v = (Matrix**)malloc(total_params * sizeof(Matrix*));
        
        // Fill parameter arrays
        int i = 0;
        layer = net->input_layer;
        while (layer) {
            if (layer->weights) {
                net->optimizer->params[i] = layer->weights;
                net->optimizer->grads[i] = layer->grad_weights;
                net->optimizer->m[i] = matrix_create(layer->weights->rows, layer->weights->cols);
                net->optimizer->v[i] = matrix_create(layer->weights->rows, layer->weights->cols);
                matrix_fill(net->optimizer->m[i], 0.0f);
                matrix_fill(net->optimizer->v[i], 0.0f);
                i++;
            }
            if (layer->biases) {
                net->optimizer->params[i] = layer->biases;
                net->optimizer->grads[i] = layer->grad_biases;
                net->optimizer->m[i] = matrix_create(layer->biases->rows, layer->biases->cols);
                net->optimizer->v[i] = matrix_create(layer->biases->rows, layer->biases->cols);
                matrix_fill(net->optimizer->m[i], 0.0f);
                matrix_fill(net->optimizer->v[i], 0.0f);
                i++;
            }
            layer = layer->next;
        }
        
        net->optimizer->param_count = i;
    }
}

void network_set_optimizer(Network* net, Optimizer* optimizer) {
    net->optimizer = optimizer;
}

Matrix* network_forward(Network* net, const Matrix* input) {
    Layer* layer = net->input_layer;
    Matrix* current_output = (Matrix*)input;  // Cast away const
    
    while (layer) {
        layer->forward(layer, current_output);
        current_output = layer->output;
        layer = layer->next;
    }
    
    // Create a copy of the output
    Matrix* output_copy = matrix_create(current_output->rows, current_output->cols);
    matrix_copy(output_copy, current_output);
    return output_copy;
}

void network_backward(Network* net, const Matrix* target) {
    // Start from output layer and move backwards
    Layer* layer = net->output_layer;
    Matrix* grad = NULL;
    
    while (layer) {
        if (layer == net->output_layer) {
            // Output layer: compute derivative of loss
            grad = matrix_create(layer->output->rows, layer->output->cols);
            // Create a copy of target for subtraction
            Matrix* target_copy = matrix_create(target->rows, target->cols);
            matrix_copy(target_copy, target);
            matrix_subtract(layer->output, target_copy);
            matrix_copy(grad, layer->output);
            matrix_free(target_copy);
        } else {
            // Hidden layer: backpropagate the gradient
            Matrix* new_grad = matrix_create(layer->output->rows, layer->output->cols);
            // This would be computed based on the next layer's backward pass
            // For now, just copy (simplified)
            matrix_copy(new_grad, grad);
            matrix_free(grad);
            grad = new_grad;
        }
        
        layer->backward(layer, grad);
        // Move to previous layer by traversing backwards through the list
        Layer* prev_layer = net->input_layer;
        while (prev_layer && prev_layer->next != layer) {
            prev_layer = prev_layer->next;
        }
        layer = prev_layer;
    }
    
    if (grad) matrix_free(grad);
}

void network_update(Network* net) {
    if (net->optimizer) {
        net->optimizer->update(net->optimizer);
    }
}

float network_train(Network* net, const Matrix* input, const Matrix* target) {
    net->is_training = 1;
    
    // Forward pass
    Matrix* output = network_forward(net, input);
    
    // Compute loss
    float loss = cross_entropy_loss(output, target);
    
    // Backward pass
    network_backward(net, target);
    
    // Update parameters
    network_update(net);
    
    matrix_free(output);
    return loss;
}

float network_test(Network* net, const Matrix* input, const Matrix* target) {
    net->is_training = 0;
    
    // Forward pass
    Matrix* output = network_forward(net, input);
    
    // Compute loss
    float loss = cross_entropy_loss(output, target);
    
    matrix_free(output);
    return loss;
}

void network_free(Network* net) {
    Layer* layer = net->input_layer;
    while (layer) {
        Layer* next = layer->next;
        layer->free(layer);
        layer = next;
    }
    
    if (net->optimizer) {
        net->optimizer->free(net->optimizer);
    }
    
    free(net);
}