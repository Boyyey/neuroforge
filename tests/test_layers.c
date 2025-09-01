#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/layers/layer.h"
#include "../src/activations/activation.h"
#include "../src/matrix.h"

void test_dense_layer_forward() {
    printf("Testing dense layer forward pass...\n");
    
    // Create a dense layer
    Layer* layer = dense_layer(3, 2, ACTIVATION_RELU);
    
    // Create input
    Matrix* input = matrix_create(1, 3);
    float input_data[] = {1.0f, 2.0f, 3.0f};
    matrix_from_array(input, input_data);
    
    // Forward pass
    layer->forward(layer, input);
    
    // Check output dimensions
    assert(layer->output != NULL);
    assert(layer->output->rows == 1);
    assert(layer->output->cols == 2);
    
    printf("Dense layer forward pass: PASSED\n");
    
    // Cleanup
    matrix_free(input);
    layer->free(layer);
}

void test_dense_layer_backward() {
    printf("Testing dense layer backward pass...\n");
    
    // Create a dense layer
    Layer* layer = dense_layer(3, 2, ACTIVATION_RELU);
    
    // Create input and do forward pass
    Matrix* input = matrix_create(1, 3);
    float input_data[] = {1.0f, 2.0f, 3.0f};
    matrix_from_array(input, input_data);
    
    layer->forward(layer, input);
    
    // Create output gradient
    Matrix* output_grad = matrix_create(1, 2);
    matrix_fill(output_grad, 1.0f);
    
    // Backward pass
    layer->backward(layer, output_grad);
    
    // Check that gradients were computed
    assert(layer->grad_weights != NULL);
    assert(layer->grad_biases != NULL);
    
    printf("Dense layer backward pass: PASSED\n");
    
    // Cleanup
    matrix_free(input);
    matrix_free(output_grad);
    layer->free(layer);
}

void test_dense_layer_update() {
    printf("Testing dense layer parameter update...\n");
    
    // Create a dense layer with no activation to ensure gradients are non-zero
    Layer* layer = dense_layer(3, 2, ACTIVATION_NONE);
    
    // Store original weights and biases
    Matrix* original_weights = matrix_create(layer->weights->rows, layer->weights->cols);
    matrix_copy(original_weights, layer->weights);
    Matrix* original_biases = matrix_create(layer->biases->rows, layer->biases->cols);
    matrix_copy(original_biases, layer->biases);
    
    // Create input and do forward pass
    Matrix* input = matrix_create(1, 3);
    float input_data[] = {1.0f, 2.0f, 3.0f};
    matrix_from_array(input, input_data);
    
    layer->forward(layer, input);
    
    // Create output gradient and do backward pass
    Matrix* output_grad = matrix_create(1, 2);
    matrix_fill(output_grad, 1.0f);
    
    layer->backward(layer, output_grad);
    
    // Update parameters
    layer->update(layer, 1.0f);  // Use larger learning rate to ensure detectable change
    
    // Check that parameters changed
    int weights_changed = 0;
    for (size_t i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
        if (layer->weights->data[i] != original_weights->data[i]) {
            weights_changed = 1;
            break;
        }
    }
    
    int biases_changed = 0;
    for (size_t i = 0; i < layer->biases->rows * layer->biases->cols; i++) {
        if (layer->biases->data[i] != original_biases->data[i]) {
            biases_changed = 1;
            break;
        }
    }
    
    assert(weights_changed);
    assert(biases_changed);
    
    printf("Dense layer parameter update: PASSED\n");
    
    // Cleanup
    matrix_free(input);
    matrix_free(output_grad);
    matrix_free(original_weights);
    matrix_free(original_biases);
    layer->free(layer);
}

void test_activation_functions() {
    printf("Testing activation functions...\n");
    
    // Create a matrix
    Matrix* m = matrix_create(2, 2);
    float data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    matrix_from_array(m, data);
    
    // Test ReLU
    activate(m, ACTIVATION_RELU);
    assert(m->data[0] == 0.0f);  // -1 -> 0
    assert(m->data[1] == 0.0f);  // 0 -> 0
    assert(m->data[2] == 1.0f);  // 1 -> 1
    assert(m->data[3] == 2.0f);  // 2 -> 2
    
    printf("Activation functions: PASSED\n");
    
    // Cleanup
    matrix_free(m);
}

void test_dropout_layer() {
    printf("Testing dropout layer...\n");
    
    // Create a dropout layer
    Layer* layer = dropout_layer(0.5f);
    
    // Create input
    Matrix* input = matrix_create(2, 2);
    matrix_fill(input, 1.0f);
    
    // Set training mode
    layer->is_training = 1;
    
    // Forward pass
    layer->forward(layer, input);
    
    // Check that some values are zeroed out
    size_t zero_count = 0;
    size_t scaled_count = 0;
    
    for (size_t i = 0; i < input->rows * input->cols; i++) {
        if (layer->output->data[i] == 0.0f) {
            zero_count++;
        } else if (layer->output->data[i] == 2.0f) {  // 1.0 * (1/(1-0.5)) = 2.0
            scaled_count++;
        }
    }
    
    // With 50% dropout, we expect some zeros and some scaled values
    assert(zero_count + scaled_count == input->rows * input->cols);
    
    // Test inference mode (no dropout)
    layer->is_training = 0;
    layer->forward(layer, input);
    
    // All values should be the same as input
    for (size_t i = 0; i < input->rows * input->cols; i++) {
        assert(layer->output->data[i] == input->data[i]);
    }
    
    printf("Dropout layer: PASSED\n");
    
    // Cleanup
    matrix_free(input);
    layer->free(layer);
}

int main() {
    printf("Running layer tests...\n\n");
    
    test_dense_layer_forward();
    test_dense_layer_backward();
    test_dense_layer_update();
    test_activation_functions();
    test_dropout_layer();
    
    printf("\nAll layer tests PASSED!\n");
    return 0;
}