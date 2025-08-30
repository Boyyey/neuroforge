#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../matrix.h"

// Activation function types
typedef enum {
    ACTIVATION_NONE,     // Linear (no activation)
    ACTIVATION_SIGMOID,  // Sigmoid function
    ACTIVATION_RELU,     // Rectified Linear Unit
    ACTIVATION_TANH,     // Hyperbolic tangent
    ACTIVATION_SOFTMAX,  // Softmax function
    ACTIVATION_LEAKY_RELU, // Leaky ReLU
    ACTIVATION_ELU,      // Exponential Linear Unit
    ACTIVATION_SELU,     // Scaled Exponential Linear Unit
    ACTIVATION_SWISH,    // Swish (self-gated) activation
    ACTIVATION_MISH,     // Mish activation
    ACTIVATION_GELU,     // Gaussian Error Linear Unit
    ACTIVATION_MAX       // Number of activation types
} ActivationType;

/**
 * @brief Apply activation function to a matrix in-place
 * 
 * @param m Matrix to apply activation to
 * @param activation Type of activation function to apply
 */
void activate(Matrix* m, ActivationType activation);

/**
 * @brief Apply derivative of activation function to gradient matrix in-place
 * 
 * @param m Input matrix (before activation)
 * @param grad Gradient matrix to modify
 * @param activation Type of activation function
 */
void activate_derivative(const Matrix* m, Matrix* grad, ActivationType activation);

/**
 * @brief Calculate cross-entropy loss between output and target
 * 
 * @param output Network output (probabilities)
 * @param target Target values (one-hot encoded)
 * @return float Cross-entropy loss value
 */
float cross_entropy_loss(const Matrix* output, const Matrix* target);

/**
 * @brief Calculate mean squared error loss between output and target
 * 
 * @param output Network output
 * @param target Target values
 * @return float MSE loss value
 */
float mse_loss(const Matrix* output, const Matrix* target);

/**
 * @brief Calculate binary cross-entropy loss (for binary classification)
 * 
 * @param output Network output (probabilities)
 * @param target Target values (0 or 1)
 * @return float Binary cross-entropy loss value
 */
float binary_cross_entropy_loss(const Matrix* output, const Matrix* target);

/**
 * @brief Get string representation of activation function
 * 
 * @param activation Activation function type
 * @return const char* String name of activation
 */
const char* activation_get_name(ActivationType activation);

/**
 * @brief Parse activation function from string
 * 
 * @param name String name of activation function
 * @return ActivationType Corresponding activation type
 */
ActivationType activation_from_string(const char* name);

// Advanced activation functions (need to be added to activate() function)

/**
 * @brief Leaky ReLU activation function
 * 
 * @param x Input value
 * @param alpha Slope for negative values (typically 0.01)
 * @return float Activated value
 */
float leaky_relu(float x, float alpha);

/**
 * @brief Exponential Linear Unit (ELU) activation function
 * 
 * @param x Input value
 * @param alpha Alpha parameter for ELU
 * @return float Activated value
 */
float elu(float x, float alpha);

/**
 * @brief Scaled Exponential Linear Unit (SELU) activation function
 * 
 * @param x Input value
 * @return float Activated value
 */
float selu(float x);

/**
 * @brief Swish activation function (self-gated)
 * 
 * @param x Input value
 * @return float Activated value
 */
float swish(float x);

/**
 * @brief Mish activation function
 * 
 * @param x Input value
 * @return float Activated value
 */
float mish(float x);

/**
 * @brief Gaussian Error Linear Unit (GELU) activation function
 * 
 * @param x Input value
 * @return float Activated value
 */
float gelu(float x);

#endif // ACTIVATION_H