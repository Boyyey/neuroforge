#include "serialization.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Network serialization format:
// [magic_number:4][version:4][layer_count:4]
// For each layer:
//   [layer_type:4][data...]

#define NN_MAGIC 0x4E4E4C31  // "NNL1"
#define NN_VERSION 1

void network_serialize(Network* net, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    uint32_t magic = NN_MAGIC;
    uint32_t version = NN_VERSION;
    uint32_t layer_count = net->layer_count;
    
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);
    fwrite(&layer_count, sizeof(uint32_t), 1, fp);
    
    // Write each layer
    Layer* layer = net->input_layer;
    while (layer) {
        // Write layer type
        uint32_t layer_type = layer->type;
        fwrite(&layer_type, sizeof(uint32_t), 1, fp);
        
        // Write layer-specific data
        switch (layer->type) {
            case LAYER_DENSE:
                // Write weights and biases
                fwrite(&layer->input_size, sizeof(int), 1, fp);
                fwrite(&layer->output_size, sizeof(int), 1, fp);
                fwrite(&layer->activation, sizeof(int), 1, fp);
                
                // Write weights matrix
                fwrite(&layer->weights->rows, sizeof(size_t), 1, fp);
                fwrite(&layer->weights->cols, sizeof(size_t), 1, fp);
                fwrite(layer->weights->data, sizeof(float), 
                      layer->weights->rows * layer->weights->cols, fp);
                
                // Write biases matrix
                fwrite(&layer->biases->rows, sizeof(size_t), 1, fp);
                fwrite(&layer->biases->cols, sizeof(size_t), 1, fp);
                fwrite(layer->biases->data, sizeof(float), 
                      layer->biases->rows * layer->biases->cols, fp);
                break;
                
            case LAYER_CONV2D:
                // Implementation for convolutional layer
                break;
                
            case LAYER_RNN:
                // Implementation for RNN layer
                break;
                
            default:
                fprintf(stderr, "Unknown layer type: %d\n", layer->type);
                break;
        }
        
        layer = layer->next;
    }
    
    fclose(fp);
}

Network* network_deserialize(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read header
    uint32_t magic, version, layer_count;
    fread(&magic, sizeof(uint32_t), 1, fp);
    fread(&version, sizeof(uint32_t), 1, fp);
    fread(&layer_count, sizeof(uint32_t), 1, fp);
    
    if (magic != NN_MAGIC) {
        fprintf(stderr, "Invalid file format: wrong magic number\n");
        fclose(fp);
        return NULL;
    }
    
    if (version != NN_VERSION) {
        fprintf(stderr, "Unsupported version: %d (expected %d)\n", version, NN_VERSION);
        fclose(fp);
        return NULL;
    }
    
    Network* net = network_create();
    
    // Read each layer
    for (uint32_t i = 0; i < layer_count; i++) {
        uint32_t layer_type;
        fread(&layer_type, sizeof(uint32_t), 1, fp);
        
        Layer* layer = NULL;
        
        switch (layer_type) {
            case LAYER_DENSE: {
                int input_size, output_size, activation;
                fread(&input_size, sizeof(int), 1, fp);
                fread(&output_size, sizeof(int), 1, fp);
                fread(&activation, sizeof(int), 1, fp);
                
                layer = dense_layer(input_size, output_size, activation);
                
                // Read weights
                size_t rows, cols;
                fread(&rows, sizeof(size_t), 1, fp);
                fread(&cols, sizeof(size_t), 1, fp);
                
                if (rows != layer->weights->rows || cols != layer->weights->cols) {
                    fprintf(stderr, "Weight matrix size mismatch\n");
                    fclose(fp);
                    network_free(net);
                    return NULL;
                }
                
                fread(layer->weights->data, sizeof(float), rows * cols, fp);
                
                // Read biases
                fread(&rows, sizeof(size_t), 1, fp);
                fread(&cols, sizeof(size_t), 1, fp);
                
                if (rows != layer->biases->rows || cols != layer->biases->cols) {
                    fprintf(stderr, "Bias matrix size mismatch\n");
                    fclose(fp);
                    network_free(net);
                    return NULL;
                }
                
                fread(layer->biases->data, sizeof(float), rows * cols, fp);
                break;
            }
                
            case LAYER_CONV2D:
                // Implementation for convolutional layer
                break;
                
            case LAYER_RNN:
                // Implementation for RNN layer
                break;
                
            default:
                fprintf(stderr, "Unknown layer type: %d\n", layer_type);
                fclose(fp);
                network_free(net);
                return NULL;
        }
        
        if (layer) {
            network_add_layer(net, layer);
        }
    }
    
    fclose(fp);
    return net;
}

void save_checkpoint(Network* net, Optimizer* opt, const char* filename) {
    // Save network
    char net_filename[256];
    snprintf(net_filename, sizeof(net_filename), "%s.net", filename);
    network_serialize(net, net_filename);
    
    // Save optimizer state (simplified implementation)
    char opt_filename[256];
    snprintf(opt_filename, sizeof(opt_filename), "%s.opt", filename);
    
    FILE* fp = fopen(opt_filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening optimizer file for writing: %s\n", opt_filename);
        return;
    }
    
    // Write optimizer type
    uint32_t name_len = strlen(opt->name);
    fwrite(&name_len, sizeof(uint32_t), 1, fp);
    fwrite(opt->name, sizeof(char), name_len, fp);
    
    // Write optimizer parameters
    fwrite(&opt->learning_rate, sizeof(float), 1, fp);
    fwrite(&opt->beta1, sizeof(float), 1, fp);
    fwrite(&opt->beta2, sizeof(float), 1, fp);
    fwrite(&opt->epsilon, sizeof(float), 1, fp);
    fwrite(&opt->t, sizeof(int), 1, fp);
    
    fclose(fp);
}

void load_checkpoint(Network** net, Optimizer** opt, const char* filename) {
    // Load network
    char net_filename[256];
    snprintf(net_filename, sizeof(net_filename), "%s.net", filename);
    *net = network_deserialize(net_filename);
    
    if (!*net) {
        fprintf(stderr, "Failed to load network from %s\n", net_filename);
        return;
    }
    
    // Load optimizer state (simplified implementation)
    char opt_filename[256];
    snprintf(opt_filename, sizeof(opt_filename), "%s.opt", filename);
    
    FILE* fp = fopen(opt_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening optimizer file for reading: %s\n", opt_filename);
        return;
    }
    
    // Read optimizer type
    uint32_t name_len;
    fread(&name_len, sizeof(uint32_t), 1, fp);
    
    char name[64];
    fread(name, sizeof(char), name_len, fp);
    name[name_len] = '\0';
    
    // Create optimizer based on type
    if (strcmp(name, "adam") == 0) {
        float learning_rate, beta1, beta2, epsilon;
        int t;
        
        fread(&learning_rate, sizeof(float), 1, fp);
        fread(&beta1, sizeof(float), 1, fp);
        fread(&beta2, sizeof(float), 1, fp);
        fread(&epsilon, sizeof(float), 1, fp);
        fread(&t, sizeof(int), 1, fp);
        
        *opt = adam_optimizer(learning_rate, beta1, beta2, epsilon);
        (*opt)->t = t;
    } else if (strcmp(name, "sgd") == 0) {
        float learning_rate, momentum;
        
        fread(&learning_rate, sizeof(float), 1, fp);
        fread(&momentum, sizeof(float), 1, fp);
        
        *opt = sgd_optimizer(learning_rate, momentum);
    }
    
    fclose(fp);
    
    // Compile network with optimizer
    network_compile(*net, *opt, 0.0f);  // Assuming no L2 regularization for now
}

// Wrapper functions to match network.h declarations
void network_save(Network* net, const char* filename) {
    network_serialize(net, filename);
}

Network* network_load(const char* filename) {
    return network_deserialize(filename);
}