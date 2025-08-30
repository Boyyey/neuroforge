#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../src/matrix.h"
#include "../src/layers/layer.h"
#include "../src/activations/activation.h"

void test_dense_layer_forward() {
    printf("Testing dense layer forward pass...\n");
    
    // Create a dense layer
    Layer* layer = dense_layer(3, 2, ACTIVATION_RELU);
    
    // Create input matrix
    Matrix* input = matrix_create(2, 3);  // Batch size 2, input size 3
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    memcpy(input->data, input_data, sizeof(input_data));
    
    // Manually set weights and biases for predictable results
    float weights_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float biases_data[] = {0.1f, 0.2f};
    memcpy(layer->weights->data, weights_data, sizeof(weights_data));
    memcpy(layer->biases->data, biases_data, sizeof(biases_data));
    
    // Perform forward pass
    layer->forward(layer, input);
    
    // Verify output
    // Expected calculations:
    // Sample 1: [1,2,3] * [[0.1,0.4], [0.2,0.5], [0.3,0.6]] + [0.1,0.2]
    // = [1*0.1+2*0.2+3*0.3+0.1, 1*0.4+2*0.5+3*0.6+0.2]
    // = [0.1+0.4+0.9+0.1, 0.4+1.0+1.8+0.2] = [1.5, 3.4]
    // After ReLU: [1.5, 3.4] (no change since positive)
    
    // Sample 2: [4,5,6] * [[0.1,0.4], [0.2,0.5], [0.3,0.6]] + [0.1,0.2]
    // = [4*0.1+5*0.2+6*0.3+0.1, 4*0.4+5*0.5+6*0.6+0.2]
    // = [0.4+1.0+1.8+0.1, 1.6+2.5+3.6+0.2] = [3.3, 7.9]
    // After ReLU: [3.3, 7.9] (no change since positive)
    
    assert(fabs(layer->output->data[0] - 1.5f) < 1e-6);
    assert(fabs(layer->output->data[1] - 3.4f) < 1e-6);
    assert(fabs(layer->output->data[2] - 3.3f) < 1e-6);
    assert(fabs(layer->output->data[3] - 7.9f) < 1e-6);
    
    // Clean up
    layer->free(layer);
    matrix_free(input);
    
    printf("Dense layer forward test passed!\n");
}

void test_dense_layer_backward() {
    printf("Testing dense layer backward pass...\n");
    
    // Create a dense layer
    Layer* layer = dense_layer(2, 3, ACTIVATION_SIGMOID);
    
    // Create input matrix
    Matrix* input = matrix_create(2, 2);  // Batch size 2, input size 2
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(input->data, input_data, sizeof(input_data));
    
    // Manually set weights and biases
    float weights_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float biases_data[] = {0.1f, 0.2f, 0.3f};
    memcpy(layer->weights->data, weights_data, sizeof(weights_data));
    memcpy(layer->biases->data, biases_data, sizeof(biases_data));
    
    // Perform forward pass first
    layer->forward(layer, input);
    
    // Create output gradient
    Matrix* output_grad = matrix_create(2, 3);  // Batch size 2, output size 3
    float grad_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    memcpy(output_grad->data, grad_data, sizeof(grad_data));
    
    // Perform backward pass
    layer->backward(layer, output_grad);
    
    // Verify that gradients were calculated (not zero)
    float weight_grad_sum = matrix_sum(layer->grad_weights);
    float bias_grad_sum = matrix_sum(layer->grad_biases);
    
    assert(fabs(weight_grad_sum) > 1e-6);
    assert(fabs(bias_grad_sum) > 1e-6);
    
    // Clean up
    layer->free(layer);
    matrix_free(input);
    matrix_free(output_grad);
    
    printf("Dense layer backward test passed!\n");
}

void test_dense_layer_update() {
    printf("Testing dense layer update...\n");
    
    // Create a dense layer
    Layer* layer = dense_layer(2, 2, ACTIVATION_RELU);
    
    // Set some gradients
    matrix_fill(layer->grad_weights, 0.1f);
    matrix_fill(layer->grad_biases, 0.2f);
    
    // Store original values
    Matrix* original_weights = matrix_copy(layer->weights);
    Matrix* original_biases = matrix_copy(layer->biases);
    
    // Perform update
    layer->update(layer, 0.1f);  // Learning rate 0.1
    
    // Verify that weights and biases were updated correctly
    // weights = weights - learning_rate * grad_weights
    // biases = biases - learning_rate * grad_biases
    for (int i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
        float expected = original_weights->data[i] - 0.1f * 0.1f;
        assert(fabs(layer->weights->data[i] - expected) < 1e-6);
    }
    
    for (int i = 0; i < layer->biases->rows * layer->biases->cols; i++) {
        float expected = original_biases->data[i] - 0.1f * 0.2f;
        assert(fabs(layer->biases->data[i] - expected) < 1e-6);
    }
    
    // Verify that gradients were reset to zero
    float weight_grad_sum = matrix_sum(layer->grad_weights);
    float bias_grad_sum = matrix_sum(layer->grad_biases);
    
    assert(fabs(weight_grad_sum) < 1e-6);
    assert(fabs(bias_grad_sum) < 1e-6);
    
    // Clean up
    layer->free(layer);
    matrix_free(original_weights);
    matrix_free(original_biases);
    
    printf("Dense layer update test passed!\n");
}

void test_activation_functions() {
    printf("Testing activation functions...\n");
    
    // Test sigmoid
    Matrix* m = matrix_create(1, 3);
    float data[] = {0.0f, 1.0f, -1.0f};
    memcpy(m->data, data, sizeof(data));
    
    activate(m, ACTIVATION_SIGMOID);
    
    // Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731, Sigmoid(-1) ≈ 0.269
    assert(fabs(m->data[0] - 0.5f) < 1e-3);
    assert(fabs(m->data[1] - 0.7310585786f) < 1e-3);
    assert(fabs(m->data[2] - 0.2689414214f) < 1e-3);
    
    // Test ReLU
    memcpy(m->data, data, sizeof(data));
    activate(m, ACTIVATION_RELU);
    
    assert(fabs(m->data[0] - 0.0f) < 1e-6);
    assert(fabs(m->data[1] - 1.0f) < 1e-6);
    assert(fabs(m->data[2] - 0.0f) < 1e-6);
    
    // Test tanh
    memcpy(m->data, data, sizeof(data));
    activate(m, ACTIVATION_TANH);
    
    // tanh(0) = 0, tanh(1) ≈ 0.7616, tanh(-1) ≈ -0.7616
    assert(fabs(m->data[0] - 0.0f) < 1e-3);
    assert(fabs(m->data[1] - 0.761594156f) < 1e-3);
    assert(fabs(m->data[2] - (-0.761594156f)) < 1e-3);
    
    matrix_free(m);
    printf("Activation functions test passed!\n");
}

void test_dropout_layer() {
    printf("Testing dropout layer...\n");
    
    // Create a dropout layer with 50% dropout rate
    Layer* layer = dropout_layer(0.5f);
    
    // Create input matrix
    Matrix* input = matrix_create(2, 3);
    matrix_fill(input, 1.0f);  // All ones
    
    // Test training mode
    layer->is_training = 1;
    layer->forward(layer, input);
    
    // In training mode with 50% dropout, approximately half the values should be zero
    // and the other half should be scaled by 2 (1/(1-0.5))
    int zero_count = 0;
    int scaled_count = 0;
    
    for (int i = 0; i < input->rows * input->cols; i++) {
        if (fabs(layer->output->data[i]) < 1e-6) {
            zero_count++;
        } else if (fabs(layer->output->data[i] - 2.0f) < 1e-6) {
            scaled_count++;
        }
    }
    
    // Should have roughly half zeros and half scaled values
    // Allow some tolerance for randomness
    assert(zero_count > 0);
    assert(scaled_count > 0);
    assert(zero_count + scaled_count == input->rows * input->cols);
    
    // Test inference mode (no dropout)
    layer->is_training = 0;
    layer->forward(layer, input);
    
    // All values should be unchanged in inference mode
    for (int i = 0; i < input->rows * input->cols; i++) {
        assert(fabs(layer->output->data[i] - 1.0f) < 1e-6);
    }
    
    // Clean up
    layer->free(layer);
    matrix_free(input);
    
    printf("Dropout layer test passed!\n");
}

int main() {
    printf("Running layer tests...\n");
    
    test_dense_layer_forward();
    test_dense_layer_backward();
    test_dense_layer_update();
    test_activation_functions();
    test_dropout_layer();
    
    printf("All layer tests passed!\n");
    return 0;
}