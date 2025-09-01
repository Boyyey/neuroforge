#include "../src/network.h"
#include "../src/layers/layer.h"
#include "../src/optimizers/optimizer.h"
#include "../src/activations/activation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("Creating Transformer network...\n");
    
    Network* net = network_create();
    
    // Build a simplified Transformer architecture
    // Input embedding layer
    network_add_layer(net, dense_layer(100, 64, ACTIVATION_NONE));  // Linear embedding
    
    // Multi-head attention layers (simplified)
    for (int i = 0; i < 3; i++) {
        // Self-attention
        network_add_layer(net, attention_layer(64, 8));  // 8 attention heads
        
        // Feed-forward network
        network_add_layer(net, dense_layer(64, 256, ACTIVATION_RELU));
        network_add_layer(net, dense_layer(256, 64, ACTIVATION_NONE));
        
        // Add dropout for regularization
        network_add_layer(net, dropout_layer(0.1f));
    }
    
    // Output projection
    network_add_layer(net, dense_layer(64, 50, ACTIVATION_SOFTMAX));
    
    // Set optimizer
    Optimizer* adam = adam_optimizer(0.001, 0.9, 0.999, 1e-8);
    network_set_optimizer(net, adam);
    
    printf("Transformer network created successfully!\n");
    printf("Architecture:\n");
    printf("  Input embedding: 100 -> 64 (linear)\n");
    printf("  3 transformer blocks:\n");
    printf("    - Self-attention (8 heads)\n");
    printf("    - Feed-forward: 64 -> 256 -> 64\n");
    printf("    - Dropout (0.1)\n");
    printf("  Output: 64 -> 50 (softmax)\n");
    
    // Create dummy training data
    printf("\nCreating dummy training data...\n");
    Matrix* train_data = matrix_create(50, 100);   // 50 sequences, 100 features each
    Matrix* train_labels = matrix_create(50, 50);  // 50 sequences, 50 classes
    
    // Fill with random data
    matrix_random_uniform(train_data, 0.0f, 1.0f);
    matrix_random_uniform(train_labels, 0.0f, 1.0f);
    
    // Normalize labels to probabilities
    for (int i = 0; i < 50; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 50; j++) {
            sum += train_labels->data[i * 50 + j];
        }
        for (int j = 0; j < 50; j++) {
            train_labels->data[i * 50 + j] /= sum;
        }
    }
    
    printf("Training data created: %zu sequences\n", train_data->rows);
    
    // Training loop (simplified)
    printf("\nStarting training...\n");
    for (int epoch = 0; epoch < 3; epoch++) {
        float loss = network_train(net, train_data, train_labels);
        printf("Epoch %d: Loss = %.4f\n", epoch, loss);
    }
    
    printf("Training completed!\n");
    
    // Save the model
    network_save(net, "transformer_model.bin");
    printf("Model saved to transformer_model.bin\n");
    
    // Cleanup
    matrix_free(train_data);
    matrix_free(train_labels);
    network_free(net);
    
    // Free optimizer (we need to implement this)
    // optimizer_free(adam);  // Commented out until we implement it
    
    printf("Transformer example completed successfully!\n");
    return 0;
}