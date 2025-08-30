#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/network.h"
#include "../src/layers/layer.h"
#include "../src/optimizers/optimizer.h"

// Simplified MNIST dataset loading (in a real implementation, you'd load from files)
void load_mnist_data(Matrix** train_images, Matrix** train_labels,
                    Matrix** test_images, Matrix** test_labels) {
    // This is a placeholder - in a real implementation, you would load
    // the actual MNIST dataset from files
    
    *train_images = matrix_create(60000, 784);  // 60,000 samples, 28x28 pixels
    *train_labels = matrix_create(60000, 10);   // 60,000 samples, 10 classes
    *test_images = matrix_create(10000, 784);   // 10,000 samples, 28x28 pixels
    *test_labels = matrix_create(10000, 10);    // 10,000 samples, 10 classes
    
    // Fill with random data for demonstration
    matrix_random_uniform(*train_images, 0.0f, 1.0f);
    matrix_random_uniform(*test_images, 0.0f, 1.0f);
    
    // Create one-hot labels
    for (int i = 0; i < 60000; i++) {
        int label = rand() % 10;
        matrix_fill(*train_labels, 0.0f);
        (*train_labels)->data[i * 10 + label] = 1.0f;
    }
    
    for (int i = 0; i < 10000; i++) {
        int label = rand() % 10;
        matrix_fill(*test_labels, 0.0f);
        (*test_labels)->data[i * 10 + label] = 1.0f;
    }
}

int main() {
    srand(time(NULL));
    
    // Load MNIST data
    Matrix* train_images, *train_labels, *test_images, *test_labels;
    load_mnist_data(&train_images, &train_labels, &test_images, &test_labels);
    
    // Create network for MNIST classification
    Network* net = network_create();
    network_add_layer(net, dense_layer(784, 128, ACTIVATION_RELU));
    network_add_layer(net, dropout_layer(0.2f));
    network_add_layer(net, dense_layer(128, 64, ACTIVATION_RELU));
    network_add_layer(net, dropout_layer(0.2f));
    network_add_layer(net, dense_layer(64, 10, ACTIVATION_SOFTMAX));
    
    // Compile with Adam optimizer
    Optimizer* adam = adam_optimizer(0.001f, 0.9f, 0.999f, 1e-8f);
    network_compile(net, adam, 0.0001f);  // L2 regularization
    
    // Training parameters
    int epochs = 10;
    int batch_size = 64;
    int num_batches = train_images->rows / batch_size;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Create batch views
            Matrix input_view = {
                .rows = batch_size,
                .cols = train_images->cols,
                .stride = train_images->stride,
                .data = &train_images->data[batch * batch_size * train_images->stride],
                .is_view = 1
            };
            
            Matrix target_view = {
                .rows = batch_size,
                .cols = train_labels->cols,
                .stride = train_labels->stride,
                .data = &train_labels->data[batch * batch_size * train_labels->stride],
                .is_view = 1
            };
            
            // Train on batch
            float loss = network_train(net, &input_view, &target_view);
            total_loss += loss;
            
            if (batch % 100 == 0) {
                printf("Epoch %d, Batch %d, Loss: %.4f\n", epoch, batch, loss);
            }
        }
        
        printf("Epoch %d, Average Loss: %.4f\n", epoch, total_loss / num_batches);
        
        // Test on validation set
        float test_loss = network_test(net, test_images, test_labels);
        printf("Epoch %d, Test Loss: %.4f\n", epoch, test_loss);
    }
    
    // Save model
    network_serialize(net, "mnist_model.bin");
    
    // Cleanup
    network_free(net);
    optimizer_free(adam);
    matrix_free(train_images);
    matrix_free(train_labels);
    matrix_free(test_images);
    matrix_free(test_labels);
    
    return 0;
}