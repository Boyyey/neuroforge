#include "../src/network.h"
#include "../src/layers/layer.h"
#include "../src/optimizers/optimizer.h"
#include "../src/activations/activation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Create network
    Network* net = network_create();
    
    // Add layers
    network_add_layer(net, dense_layer(2, 4, ACTIVATION_RELU));
    network_add_layer(net, dense_layer(4, 1, ACTIVATION_SIGMOID));
    
    // Set optimizer
    Optimizer* adam = adam_optimizer(0.01, 0.9, 0.999, 1e-8);
    network_set_optimizer(net, adam);
    
    // Create training data (XOR problem)
    Matrix* inputs = matrix_create(4, 2);
    Matrix* targets = matrix_create(4, 1);
    
    // XOR inputs: [0,0], [0,1], [1,0], [1,1]
    float input_data[] = {0,0, 0,1, 1,0, 1,1};
    float target_data[] = {0, 1, 1, 0};
    
    matrix_from_array(inputs, input_data);
    matrix_from_array(targets, target_data);
    
    printf("Training XOR network...\n");
    
    // Training loop
    for (int epoch = 0; epoch < 1000; epoch++) {
        float loss = network_train(net, inputs, targets);
        
        if (epoch % 100 == 0) {
            printf("Epoch %d: Loss = %.4f\n", epoch, loss);
        }
    }
    
    // Test the network
    Matrix* output = network_forward(net, inputs);
    printf("\nFinal predictions:\n");
    matrix_print(output, "Output");
    
    // Save the model
    network_save(net, "xor_model.bin");
    printf("Model saved to xor_model.bin\n");
    
    // Cleanup
    matrix_free(inputs);
    matrix_free(targets);
    matrix_free(output);
    network_free(net);
    
    // Free optimizer (we need to implement this)
    // optimizer_free(adam);  // Commented out until we implement it
    
    return 0;
}