# NeuroForge Usage Guide

## Quick Start

### 1. Building the Library

#### On Windows (MinGW):
```bash
# Use the provided batch file
build.bat

# Or manually with make
mingw32-make
```

#### On Unix/Linux/macOS:
```bash
# Using make
make

# Using CMake (recommended)
mkdir build && cd build
cmake ..
make -j$(nproc)

# Using the install script
./install.sh
```

### 2. Using in Your Project

#### Include the Headers
```c
#include "neuroforge/network.h"
#include "neuroforge/layers/layer.h"
#include "neuroforge/optimizers/optimizer.h"
#include "neuroforge/activations/activation.h"
```

#### Link Against the Library
```bash
# Static linking
gcc your_file.c -L./bin -lneuroforge -lm -fopenmp

# With pkg-config (if installed)
gcc your_file.c $(pkg-config --cflags --libs neuroforge)
```

## Basic Examples

### Simple Feedforward Network

```c
#include "neuroforge/network.h"
#include "neuroforge/layers/layer.h"
#include "neuroforge/optimizers/optimizer.h"
#include <stdio.h>

int main() {
    // Create network
    Network* net = network_create();
    
    // Add layers
    network_add_layer(net, dense_layer(2, 4, ACTIVATION_RELU));
    network_add_layer(net, dense_layer(4, 1, ACTIVATION_SIGMOID));
    
    // Set optimizer
    network_set_optimizer(net, adam_optimizer(0.01));
    
    // Create training data (XOR problem)
    Matrix* input = matrix_create(4, 2);
    Matrix* target = matrix_create(4, 1);
    
    // XOR inputs: [0,0], [0,1], [1,0], [1,1]
    float input_data[] = {0,0, 0,1, 1,0, 1,1};
    float target_data[] = {0, 1, 1, 0};
    
    matrix_from_array(input, input_data);
    matrix_from_array(target, target_data);
    
    // Training loop
    for (int epoch = 0; epoch < 1000; epoch++) {
        float loss = network_train(net, input, target);
        
        if (epoch % 100 == 0) {
            printf("Epoch %d: Loss = %.4f\n", epoch, loss);
        }
    }
    
    // Test the network
    Matrix* output = network_forward(net, input);
    printf("\nFinal predictions:\n");
    matrix_print(output, "Output");
    
    // Cleanup
    matrix_free(input);
    matrix_free(target);
    matrix_free(output);
    network_free(net);
    
    return 0;
}
```

### MNIST Classification

```c
#include "neuroforge/network.h"
#include "neuroforge/layers/layer.h"
#include <stdio.h>

int main() {
    Network* net = network_create();
    
    // Build CNN architecture
    network_add_layer(net, conv2d_layer(1, 32, 3, 1, 1, ACTIVATION_RELU));
    network_add_layer(net, conv2d_layer(32, 64, 3, 1, 1, ACTIVATION_RELU));
    network_add_layer(net, dense_layer(7*7*64, 128, ACTIVATION_RELU));
    network_add_layer(net, dense_layer(128, 10, ACTIVATION_SOFTMAX));
    
    // Set optimizer
    network_set_optimizer(net, adam_optimizer(0.001));
    
    // Load MNIST data (you'll need to implement this)
    // Matrix* train_data = load_mnist_images("train-images-idx3-ubyte");
    // Matrix* train_labels = load_mnist_labels("train-labels-idx1-ubyte");
    
    // Training loop
    // for (int epoch = 0; epoch < 10; epoch++) {
    //     float loss = network_train(net, train_data, train_labels);
    //     printf("Epoch %d: Loss = %.4f\n", epoch, loss);
    // }
    
    // Save the model
    network_save(net, "mnist_model.bin");
    
    network_free(net);
    return 0;
}
```

## Advanced Usage

### Custom Layer Implementation

```c
#include "neuroforge/layers/layer.h"

// Custom activation function
void custom_activation(Matrix* input, Matrix* output) {
    for (size_t i = 0; i < input->rows * input->cols; i++) {
        float x = input->data[i];
        output->data[i] = x * x; // Square activation
    }
}

// Custom layer forward pass
void custom_forward(Layer* layer, const Matrix* input) {
    // Store input
    if (layer->input) matrix_free(layer->input);
    layer->input = matrix_copy(input);
    
    // Apply custom transformation
    if (!layer->output) {
        layer->output = matrix_create(input->rows, input->cols);
    }
    
    custom_activation(input, layer->output);
}

// Create custom layer
Layer* custom_layer(int input_size, int output_size) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    memset(layer, 0, sizeof(Layer));
    
    layer->type = LAYER_CUSTOM;
    strcpy(layer->name, "custom_layer");
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    // Set method pointers
    layer->forward = custom_forward;
    layer->backward = NULL; // Implement if needed
    layer->update = NULL;   // Implement if needed
    layer->free = NULL;     // Implement if needed
    
    return layer;
}
```

### Model Serialization

```c
#include "neuroforge/serialization.h"

// Save model
void save_model(Network* net, const char* filename) {
    if (network_save(net, filename) == 0) {
        printf("Model saved to %s\n", filename);
    } else {
        printf("Failed to save model\n");
    }
}

// Load model
Network* load_model(const char* filename) {
    Network* net = network_load(filename);
    if (net) {
        printf("Model loaded from %s\n", filename);
        return net;
    } else {
        printf("Failed to load model\n");
        return NULL;
    }
}
```

### GPU Acceleration (CUDA)

```c
#ifdef USE_CUDA
#include "neuroforge/cuda/cuda_ops.h"

void enable_gpu_acceleration(Network* net) {
    // Move all matrices to GPU
    Layer* layer = net->input_layer;
    while (layer) {
        if (layer->weights) {
            matrix_to_gpu(layer->weights);
        }
        if (layer->biases) {
            matrix_to_gpu(layer->biases);
        }
        layer = layer->next;
    }
    printf("GPU acceleration enabled\n");
}
#endif
```

## Performance Tips

### 1. Batch Processing
```c
// Process multiple samples at once for better performance
Matrix* batch_input = matrix_create(batch_size * input_size, 1);
Matrix* batch_target = matrix_create(batch_size * output_size, 1);

// Fill batch data
// ... fill batch_input and batch_target ...

// Train on batch
float loss = network_train(net, batch_input, batch_target);
```

### 2. Memory Management
```c
// Use matrix views for zero-copy operations
Matrix* view = matrix_view(parent_matrix, row_start, col_start, rows, cols);

// Don't forget to free views
matrix_free(view);
```

### 3. Optimizer Tuning
```c
// Adam optimizer with custom parameters
Optimizer* adam = adam_optimizer(0.001);  // learning rate
adam->beta1 = 0.9;                        // momentum
adam->beta2 = 0.999;                      // RMS decay
adam->epsilon = 1e-8;                     // numerical stability
```

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Ensure all dependencies are installed
   - Check compiler version compatibility
   - Verify include paths are correct

2. **Runtime Errors**
   - Check matrix dimensions match
   - Verify memory allocation
   - Use debug builds for detailed error messages

3. **Performance Issues**
   - Enable OpenMP for parallelization
   - Use CUDA if available
   - Profile with tools like gprof or valgrind

### Debug Mode

```bash
# Build with debug information
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Or with make
make CFLAGS="-g -O0 -DDEBUG"
```

## Getting Help

- Check the examples in the `examples/` directory
- Run tests to verify your installation
- Look at the source code for implementation details
- Open an issue on GitHub for bugs or questions

---

**Happy coding with NeuroForge! 🚀**
