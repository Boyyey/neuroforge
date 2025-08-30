#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../src/matrix.h"
#include "../src/optimizers/optimizer.h"

void test_sgd_optimizer() {
    printf("Testing SGD optimizer...\n");
    
    // Create a simple parameter and gradient
    Matrix* param = matrix_create(2, 2);
    Matrix* grad = matrix_create(2, 2);
    
    matrix_fill(param, 1.0f);  // Initial parameter values
    matrix_fill(grad, 0.1f);   // Gradient values
    
    // Create SGD optimizer
    Optimizer* optimizer = sgd_optimizer(0.1f, 0.0f);  // Learning rate 0.1, no momentum
    
    // Set up optimizer parameters
    optimizer->param_count = 1;
    optimizer->params = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(sizeof(Matrix*));
    
    optimizer->params[0] = param;
    optimizer->grads[0] = grad;
    
    // Store original parameter values
    Matrix* original_param = matrix_copy(param);
    
    // Perform optimization step
    optimizer->update(optimizer);
    
    // Verify parameter update: param = param - learning_rate * grad
    for (int i = 0; i < param->rows * param->cols; i++) {
        float expected = original_param->data[i] - 0.1f * 0.1f;
        assert(fabs(param->data[i] - expected) < 1e-6);
    }
    
    // Verify gradients were reset to zero
    float grad_sum = matrix_sum(grad);
    assert(fabs(grad_sum) < 1e-6);
    
    // Clean up
    optimizer->free(optimizer);
    matrix_free(param);
    matrix_free(grad);
    matrix_free(original_param);
    
    printf("SGD optimizer test passed!\n");
}

void test_sgd_optimizer_with_momentum() {
    printf("Testing SGD optimizer with momentum...\n");
    
    // Create a simple parameter and gradient
    Matrix* param = matrix_create(2, 2);
    Matrix* grad = matrix_create(2, 2);
    
    matrix_fill(param, 1.0f);  // Initial parameter values
    matrix_fill(grad, 0.1f);   // Gradient values
    
    // Create SGD optimizer with momentum
    Optimizer* optimizer = sgd_optimizer(0.1f, 0.9f);  // Learning rate 0.1, momentum 0.9
    
    // Set up optimizer parameters
    optimizer->param_count = 1;
    optimizer->params = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(sizeof(Matrix*));
    
    optimizer->params[0] = param;
    optimizer->grads[0] = grad;
    
    // Store original parameter values
    Matrix* original_param = matrix_copy(param);
    
    // Perform multiple optimization steps to test momentum
    for (int i = 0; i < 3; i++) {
        // Reset gradient to the same value each time
        matrix_fill(grad, 0.1f);
        
        // Perform optimization step
        optimizer->update(optimizer);
    }
    
    // With momentum, the update should be larger than without
    // We can't easily calculate the exact value due to momentum implementation,
    // but we can verify that the parameters changed
    float param_change = 0.0f;
    for (int i = 0; i < param->rows * param->cols; i++) {
        param_change += fabs(param->data[i] - original_param->data[i]);
    }
    
    assert(param_change > 0.1f);  // Should have changed significantly
    
    // Clean up
    optimizer->free(optimizer);
    matrix_free(param);
    matrix_free(grad);
    matrix_free(original_param);
    
    printf("SGD optimizer with momentum test passed!\n");
}

void test_adam_optimizer() {
    printf("Testing Adam optimizer...\n");
    
    // Create a simple parameter and gradient
    Matrix* param = matrix_create(2, 2);
    Matrix* grad = matrix_create(2, 2);
    
    matrix_fill(param, 1.0f);  // Initial parameter values
    matrix_fill(grad, 0.1f);   // Gradient values
    
    // Create Adam optimizer
    Optimizer* optimizer = adam_optimizer(0.001f, 0.9f, 0.999f, 1e-8f);
    
    // Set up optimizer parameters
    optimizer->param_count = 1;
    optimizer->params = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(sizeof(Matrix*));
    
    optimizer->params[0] = param;
    optimizer->grads[0] = grad;
    
    // Initialize moment vectors for Adam
    optimizer->m = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->v = (Matrix**)malloc(sizeof(Matrix*));
    
    optimizer->m[0] = matrix_create(2, 2);
    optimizer->v[0] = matrix_create(2, 2);
    
    matrix_fill(optimizer->m[0], 0.0f);
    matrix_fill(optimizer->v[0], 0.0f);
    
    // Store original parameter values
    Matrix* original_param = matrix_copy(param);
    
    // Perform optimization step
    optimizer->update(optimizer);
    
    // Verify that parameters were updated
    float param_change = 0.0f;
    for (int i = 0; i < param->rows * param->cols; i++) {
        param_change += fabs(param->data[i] - original_param->data[i]);
    }
    
    assert(param_change > 1e-6);  // Should have changed
    
    // Verify gradients were reset to zero
    float grad_sum = matrix_sum(grad);
    assert(fabs(grad_sum) < 1e-6);
    
    // Verify moment vectors were updated
    float m_sum = matrix_sum(optimizer->m[0]);
    float v_sum = matrix_sum(optimizer->v[0]);
    
    assert(fabs(m_sum) > 1e-6);  // Should have been updated
    assert(fabs(v_sum) > 1e-6);  // Should have been updated
    
    // Clean up
    optimizer->free(optimizer);
    matrix_free(param);
    matrix_free(grad);
    matrix_free(original_param);
    
    printf("Adam optimizer test passed!\n");
}

void test_adam_optimizer_multiple_steps() {
    printf("Testing Adam optimizer with multiple steps...\n");
    
    // Create a simple parameter and gradient
    Matrix* param = matrix_create(2, 2);
    Matrix* grad = matrix_create(2, 2);
    
    matrix_fill(param, 1.0f);  // Initial parameter values
    matrix_fill(grad, 0.1f);   // Gradient values
    
    // Create Adam optimizer
    Optimizer* optimizer = adam_optimizer(0.001f, 0.9f, 0.999f, 1e-8f);
    
    // Set up optimizer parameters
    optimizer->param_count = 1;
    optimizer->params = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(sizeof(Matrix*));
    
    optimizer->params[0] = param;
    optimizer->grads[0] = grad;
    
    // Initialize moment vectors for Adam
    optimizer->m = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->v = (Matrix**)malloc(sizeof(Matrix*));
    
    optimizer->m[0] = matrix_create(2, 2);
    optimizer->v[0] = matrix_create(2, 2);
    
    matrix_fill(optimizer->m[0], 0.0f);
    matrix_fill(optimizer->v[0], 0.0f);
    
    // Store original parameter values
    Matrix* original_param = matrix_copy(param);
    
    // Perform multiple optimization steps
    for (int i = 0; i < 5; i++) {
        // Reset gradient to the same value each time
        matrix_fill(grad, 0.1f);
        
        // Perform optimization step
        optimizer->update(optimizer);
    }
    
    // Verify that parameters were updated
    float param_change = 0.0f;
    for (int i = 0; i < param->rows * param->cols; i++) {
        param_change += fabs(param->data[i] - original_param->data[i]);
    }
    
    assert(param_change > 1e-6);  // Should have changed
    
    // Verify time step was incremented
    assert(optimizer->t == 5);
    
    // Clean up
    optimizer->free(optimizer);
    matrix_free(param);
    matrix_free(grad);
    matrix_free(original_param);
    
    printf("Adam optimizer multiple steps test passed!\n");
}

void test_optimizer_with_multiple_parameters() {
    printf("Testing optimizer with multiple parameters...\n");
    
    // Create multiple parameters and gradients
    Matrix* param1 = matrix_create(2, 2);
    Matrix* param2 = matrix_create(3, 1);
    Matrix* grad1 = matrix_create(2, 2);
    Matrix* grad2 = matrix_create(3, 1);
    
    matrix_fill(param1, 1.0f);
    matrix_fill(param2, 2.0f);
    matrix_fill(grad1, 0.1f);
    matrix_fill(grad2, 0.2f);
    
    // Create SGD optimizer
    Optimizer* optimizer = sgd_optimizer(0.1f, 0.0f);
    
    // Set up optimizer parameters
    optimizer->param_count = 2;
    optimizer->params = (Matrix**)malloc(2 * sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(2 * sizeof(Matrix*));
    
    optimizer->params[0] = param1;
    optimizer->params[1] = param2;
    optimizer->grads[0] = grad1;
    optimizer->grads[1] = grad2;
    
    // Store original parameter values
    Matrix* original_param1 = matrix_copy(param1);
    Matrix* original_param2 = matrix_copy(param2);
    
    // Perform optimization step
    optimizer->update(optimizer);
    
    // Verify parameter updates
    for (int i = 0; i < param1->rows * param1->cols; i++) {
        float expected = original_param1->data[i] - 0.1f * 0.1f;
        assert(fabs(param1->data[i] - expected) < 1e-6);
    }
    
    for (int i = 0; i < param2->rows * param2->cols; i++) {
        float expected = original_param2->data[i] - 0.1f * 0.2f;
        assert(fabs(param2->data[i] - expected) < 1e-6);
    }
    
    // Verify gradients were reset to zero
    float grad1_sum = matrix_sum(grad1);
    float grad2_sum = matrix_sum(grad2);
    
    assert(fabs(grad1_sum) < 1e-6);
    assert(fabs(grad2_sum) < 1e-6);
    
    // Clean up
    optimizer->free(optimizer);
    matrix_free(param1);
    matrix_free(param2);
    matrix_free(grad1);
    matrix_free(grad2);
    matrix_free(original_param1);
    matrix_free(original_param2);
    
    printf("Optimizer with multiple parameters test passed!\n");
}

int main() {
    printf("Running optimizer tests...\n");
    
    test_sgd_optimizer();
    test_sgd_optimizer_with_momentum();
    test_adam_optimizer();
    test_adam_optimizer_multiple_steps();
    test_optimizer_with_multiple_parameters();
    
    printf("All optimizer tests passed!\n");
    return 0;
}