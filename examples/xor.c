#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/network.h"
#include "../src/layers/layer.h"
#include "../src/optimizers/optimizer.h"

int main() {
    // Create XOR dataset
    Matrix* inputs = matrix_create(4, 2);
    Matrix* targets = matrix_create(4, 1);
    
    float input_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float target_data[] = {0, 1, 1, 0};
    
    memcpy(inputs->data, input_data, sizeof(input_data));
    memcpy(targets->data, target_data, sizeof(target_data));
    
    // Create network
    Network* net = network_create();
    network_add_layer(net, dense_layer(2, 4, ACTIVATION_RELU));
    network_add_layer(net, dense_layer(4, 1, ACTIVATION_SIGMOID));
    
    // Compile with Adam optimizer
    Optimizer* adam = adam_optimizer(0.01f, 0.9f, 0.999f, 1e-8f);
    network_compile(net, adam, 0.0f);  // No L2 regularization
    
    // Train network
    for (int epoch = 0; epoch < 10000; epoch++) {
        float loss = network_train(net, inputs, targets);
        
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %.4f\n", epoch, loss);
        }
    }
    
    // Test network
    printf("\nTesting XOR network:\n");
    for (int i = 0; i < 4; i++) {
        Matrix input_view = {
            .rows = 1,
            .cols = 2,
            .stride = 2,
            .data = &inputs->data[i * 2],
            .is_view = 1
        };
        
        Matrix* output = network_forward(net, &input_view);
        printf("Input: [%.0f, %.0f], Output: %.4f, Expected: %.0f\n",
               input_view.data[0], input_view.data[1], 
               output->data[0], targets->data[i]);
        matrix_free(output);
    }
    
    // Save model
    network_serialize(net, "xor_model.bin");
    
    // Cleanup
    network_free(net);
    optimizer_free(adam);
    matrix_free(inputs);
    matrix_free(targets);
    
    return 0;
}