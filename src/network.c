#include "network.h"
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
    
    // Collect all parameters for the optimizer
    int param_count = 0;
    Layer* layer = net->input_layer;
    while (layer) {
        if (layer->weights) param_count++;
        if (layer->biases) param_count++;
        layer = layer->next;
    }
    
    net->optimizer->param_count = param_count;
    net->optimizer->params = (Matrix**)malloc(param_count * sizeof(Matrix*));
    net->optimizer->grads = (Matrix**)malloc(param_count * sizeof(Matrix*));
    
    int idx = 0;
    layer = net->input_layer;
    while (layer) {
        if (layer->weights) {
            net->optimizer->params[idx] = layer->weights;
            net->optimizer->grads[idx] = layer->grad_weights;
            idx++;
        }
        if (layer->biases) {
            net->optimizer->params[idx] = layer->biases;
            net->optimizer->grads[idx] = layer->grad_biases;
            idx++;
        }
        layer = layer->next;
    }
    
    // Initialize Adam moment vectors if needed
    if (strcmp(net->optimizer->name, "adam") == 0) {
        net->optimizer->m = (Matrix**)malloc(param_count * sizeof(Matrix*));
        net->optimizer->v = (Matrix**)malloc(param_count * sizeof(Matrix*));
        
        for (int i = 0; i < param_count; i++) {
            Matrix* param = net->optimizer->params[i];
            net->optimizer->m[i] = matrix_create(param->rows, param->cols);
            net->optimizer->v[i] = matrix_create(param->rows, param->cols);
            matrix_fill(net->optimizer->m[i], 0.0f);
            matrix_fill(net->optimizer->v[i], 0.0f);
        }
    }
}

Matrix* network_forward(Network* net, const Matrix* input) {
    Layer* layer = net->input_layer;
    Matrix* current_output = (Matrix*)input;  // Cast away const
    
    while (layer) {
        layer->forward(layer, current_output);
        current_output = layer->output;
        layer = layer->next;
    }
    
    return matrix_copy(current_output);
}

void network_backward(Network* net, const Matrix* target) {
    // Start from output layer and move backwards
    Layer* layer = net->output_layer;
    Matrix* grad = NULL;
    
    while (layer) {
        if (layer == net->output_layer) {
            // Output layer: compute derivative of loss
            grad = matrix_create(layer->output->rows, layer->output->cols);
            matrix_subtract(layer->output, target, grad);
        } else {
            // Hidden layer: backpropagate the gradient
            Layer* next = layer->next;
            Matrix* new_grad = matrix_create(layer->output->rows, layer->output->cols);
            // This would be computed based on the next layer's backward pass
            // For now, just copy (simplified)
            matrix_copy(new_grad, grad);
            matrix_free(grad);
            grad = new_grad;
        }
        
        layer->backward(layer, grad);
        layer = layer->prev;
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