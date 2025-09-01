#include "../src/network.h"
#include "../src/layers/layer.h"
#include "../src/optimizers/optimizer.h"
#include "../src/activations/activation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("Creating MNIST CNN network...\n");
    
    Network* net = network_create();
    
    // Build CNN architecture for MNIST (28x28 images)
    // Input: 1 channel, 28x28
    // Conv1: 1 -> 32 channels, 3x3 kernel
    network_add_layer(net, conv2d_layer(1, 32, 3, 1, 1, ACTIVATION_RELU));
    
    // Conv2: 32 -> 64 channels, 3x3 kernel
    network_add_layer(net, conv2d_layer(32, 64, 3, 1, 1, ACTIVATION_RELU));
    
    // Dense layers
    // After 2 conv layers with padding=1, size is still 28x28
    // After pooling (not implemented yet), size becomes 14x14
    // So: 14 * 14 * 64 = 12544
    network_add_layer(net, dense_layer(12544, 128, ACTIVATION_RELU));
    network_add_layer(net, dense_layer(128, 64, ACTIVATION_RELU));
    network_add_layer(net, dense_layer(64, 10, ACTIVATION_SOFTMAX));
    
    // Set optimizer
    Optimizer* adam = adam_optimizer(0.001, 0.9, 0.999, 1e-8);
    network_set_optimizer(net, adam);
    
    printf("Network created successfully!\n");
    printf("Architecture:\n");
    printf("  Input: 1x28x28\n");
    printf("  Conv1: 32x28x28 (3x3 kernel, padding=1)\n");
    printf("  Conv2: 64x28x28 (3x3 kernel, padding=1)\n");
    printf("  Dense1: 128 neurons\n");
    printf("  Dense2: 64 neurons\n");
    printf("  Output: 10 neurons (softmax)\n");
    
    // Create dummy training data (in real implementation, load MNIST)
    printf("\nCreating dummy training data...\n");
    Matrix* train_data = matrix_create(100, 784);  // 100 samples, 28*28=784 features
    Matrix* train_labels = matrix_create(100, 10); // 100 samples, 10 classes
    
    // Fill with random data for demonstration
    matrix_random_uniform(train_data, 0.0f, 1.0f);
    matrix_random_uniform(train_labels, 0.0f, 1.0f);
    
    // Normalize labels to probabilities
    for (int i = 0; i < 100; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 10; j++) {
            sum += train_labels->data[i * 10 + j];
        }
        for (int j = 0; j < 10; j++) {
            train_labels->data[i * 10 + j] /= sum;
        }
    }
    
    printf("Training data created: %zu samples\n", train_data->rows);
    
    // Training loop (simplified)
    printf("\nStarting training...\n");
    for (int epoch = 0; epoch < 5; epoch++) {
        float loss = network_train(net, train_data, train_labels);
        printf("Epoch %d: Loss = %.4f\n", epoch, loss);
    }
    
    printf("Training completed!\n");
    
    // Save the model
    network_save(net, "mnist_model.bin");
    printf("Model saved to mnist_model.bin\n");
    
    // Cleanup
    matrix_free(train_data);
    matrix_free(train_labels);
    network_free(net);
    
    // Free optimizer (we need to implement this)
    // optimizer_free(adam);  // Commented out until we implement it
    
    printf("MNIST example completed successfully!\n");
    return 0;
}