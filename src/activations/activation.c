#include "activation.h"
#include <math.h>
#include <string.h>

// Activation function names
static const char* activation_names[] = {
    "none", "sigmoid", "relu", "tanh", "softmax", 
    "leaky_relu", "elu", "selu", "swish", "mish", "gelu"
};

void activate(Matrix* m, ActivationType activation) {
    switch (activation) {
        case ACTIVATION_SIGMOID:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                m->data[i] = 1.0f / (1.0f + expf(-m->data[i]));
            }
            break;
            
        case ACTIVATION_RELU:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                m->data[i] = m->data[i] > 0 ? m->data[i] : 0;
            }
            break;
            
        case ACTIVATION_TANH:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                m->data[i] = tanhf(m->data[i]);
            }
            break;
            
        case ACTIVATION_SOFTMAX: {
            // For numerical stability, subtract the max value
            for (size_t row = 0; row < m->rows; row++) {
                float* row_data = &m->data[row * m->stride];
                float max_val = row_data[0];
                
                for (size_t col = 1; col < m->cols; col++) {
                    if (row_data[col] > max_val) {
                        max_val = row_data[col];
                    }
                }
                
                float sum = 0;
                for (size_t col = 0; col < m->cols; col++) {
                    row_data[col] = expf(row_data[col] - max_val);
                    sum += row_data[col];
                }
                
                for (size_t col = 0; col < m->cols; col++) {
                    row_data[col] /= sum;
                }
            }
            break;
        }
            
        case ACTIVATION_LEAKY_RELU:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                m->data[i] = m->data[i] > 0 ? m->data[i] : 0.01f * m->data[i];
            }
            break;
            
        case ACTIVATION_ELU:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                m->data[i] = m->data[i] > 0 ? m->data[i] : 1.0f * (expf(m->data[i]) - 1);
            }
            break;
            
        case ACTIVATION_SELU: {
            float scale = 1.0507009873554804934193349852946f;  // λ
            float alpha = 1.6732632423543772848170429916717f;  // α
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                m->data[i] = m->data[i] > 0 ? scale * m->data[i] : scale * alpha * (expf(m->data[i]) - 1);
            }
            break;
        }
            
        case ACTIVATION_SWISH:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                m->data[i] = m->data[i] / (1.0f + expf(-m->data[i]));
            }
            break;
            
        case ACTIVATION_MISH:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                float x = m->data[i];
                m->data[i] = x * tanhf(logf(1.0f + expf(x)));
            }
            break;
            
        case ACTIVATION_GELU:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                float x = m->data[i];
                m->data[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
            }
            break;
            
        case ACTIVATION_NONE:
        default:
            // No activation applied
            break;
    }
}

void activate_derivative(const Matrix* m, Matrix* grad, ActivationType activation) {
    switch (activation) {
        case ACTIVATION_SIGMOID:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                float s = 1.0f / (1.0f + expf(-m->data[i]));
                grad->data[i] *= s * (1 - s);
            }
            break;
            
        case ACTIVATION_RELU:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                grad->data[i] *= m->data[i] > 0 ? 1 : 0;
            }
            break;
            
        case ACTIVATION_TANH:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                float t = tanhf(m->data[i]);
                grad->data[i] *= 1 - t * t;
            }
            break;
            
        case ACTIVATION_LEAKY_RELU:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                grad->data[i] *= m->data[i] > 0 ? 1 : 0.01f;
            }
            break;
            
        case ACTIVATION_ELU:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                grad->data[i] *= m->data[i] > 0 ? 1 : 1.0f * expf(m->data[i]);
            }
            break;
            
        case ACTIVATION_SELU: {
            float scale = 1.0507009873554804934193349852946f;
            float alpha = 1.6732632423543772848170429916717f;
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                grad->data[i] *= m->data[i] > 0 ? scale : scale * alpha * expf(m->data[i]);
            }
            break;
        }
            
        case ACTIVATION_SWISH:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                float x = m->data[i];
                float sigmoid = 1.0f / (1.0f + expf(-x));
                grad->data[i] *= sigmoid + x * sigmoid * (1 - sigmoid);
            }
            break;
            
        case ACTIVATION_MISH:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                float x = m->data[i];
                float omega = 4.0f * (x + 1) + 4.0f * expf(2.0f * x) + expf(3.0f * x) + expf(x) * (4.0f * x + 6.0f);
                float delta = 2.0f * expf(x) + expf(2.0f * x) + 2.0f;
                float derivative = expf(x) * omega / (delta * delta);
                grad->data[i] *= derivative;
            }
            break;
            
        case ACTIVATION_GELU:
            for (size_t i = 0; i < m->rows * m->cols; i++) {
                float x = m->data[i];
                float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
                float pdf = expf(-0.5f * x * x) / sqrtf(2.0f * M_PI);
                float derivative = cdf + x * pdf;
                grad->data[i] *= derivative;
            }
            break;
            
        case ACTIVATION_SOFTMAX:
            // For softmax, the derivative is usually combined with cross-entropy loss
            // So we don't implement it separately here
            break;
            
        case ACTIVATION_NONE:
        default:
            // No derivative applied
            break;
    }
}

float cross_entropy_loss(const Matrix* output, const Matrix* target) {
    float loss = 0;
    for (size_t i = 0; i < output->rows; i++) {
        for (size_t j = 0; j < output->cols; j++) {
            size_t idx = i * output->stride + j;
            loss += -target->data[idx] * logf(output->data[idx] + 1e-10f);
        }
    }
    return loss / output->rows;
}

float mse_loss(const Matrix* output, const Matrix* target) {
    float loss = 0;
    for (size_t i = 0; i < output->rows * output->cols; i++) {
        float diff = output->data[i] - target->data[i];
        loss += diff * diff;
    }
    return loss / (output->rows * output->cols);
}

float binary_cross_entropy_loss(const Matrix* output, const Matrix* target) {
    float loss = 0;
    for (size_t i = 0; i < output->rows * output->cols; i++) {
        float y = target->data[i];
        float y_hat = output->data[i];
        loss += -y * logf(y_hat + 1e-10f) - (1 - y) * logf(1 - y_hat + 1e-10f);
    }
    return loss / (output->rows * output->cols);
}

const char* activation_get_name(ActivationType activation) {
    if (activation >= 0 && activation < ACTIVATION_MAX) {
        return activation_names[activation];
    }
    return "unknown";
}

ActivationType activation_from_string(const char* name) {
    for (int i = 0; i < ACTIVATION_MAX; i++) {
        if (strcmp(name, activation_names[i]) == 0) {
            return (ActivationType)i;
        }
    }
    return ACTIVATION_NONE;
}

// Advanced activation functions

float leaky_relu(float x, float alpha) {
    return x > 0 ? x : alpha * x;
}

float elu(float x, float alpha) {
    return x > 0 ? x : alpha * (expf(x) - 1);
}

float selu(float x) {
    float scale = 1.0507009873554804934193349852946f;
    float alpha = 1.6732632423543772848170429916717f;
    return x > 0 ? scale * x : scale * alpha * (expf(x) - 1);
}

float swish(float x) {
    return x / (1.0f + expf(-x));
}

float mish(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}