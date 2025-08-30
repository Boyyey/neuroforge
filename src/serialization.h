#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "network.h"

// Model serialization
void network_serialize(Network* net, const char* filename);
Network* network_deserialize(const char* filename);

// Checkpointing
void save_checkpoint(Network* net, Optimizer* opt, const char* filename);
void load_checkpoint(Network** net, Optimizer** opt, const char* filename);

// Training history
typedef struct {
    float* train_loss;
    float* val_loss;
    float* train_accuracy;
    float* val_accuracy;
    int epoch_count;
} TrainingHistory;

void save_training_history(TrainingHistory* history, const char* filename);
TrainingHistory* load_training_history(const char* filename);

#endif // SERIALIZATION_H