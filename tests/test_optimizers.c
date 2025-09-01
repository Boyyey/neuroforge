#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/optimizers/optimizer.h"
#include "../src/matrix.h"

void test_sgd_optimizer() {
    printf("Testing SGD optimizer...\n");
    
    // Create an SGD optimizer
    Optimizer* optimizer = sgd_optimizer(0.01f, 0.0f);
    
    // Create a parameter matrix
    Matrix* param = matrix_create(2, 2);
    matrix_fill(param, 1.0f);
    
    // Create a gradient matrix
    Matrix* grad = matrix_create(2, 2);
    matrix_fill(grad, 0.1f);
    
    // Set up optimizer
    optimizer->params = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->params[0] = param;
    optimizer->grads[0] = grad;
    optimizer->param_count = 1;
    
    // Store original parameter values
    Matrix* original_param = matrix_create(param->rows, param->cols);
    matrix_copy(original_param, param);
    
    // Run optimizer update
    optimizer->update(optimizer);
    
    // Check that parameters were updated
    int param_changed = 0;
    for (size_t i = 0; i < param->rows * param->cols; i++) {
        if (param->data[i] != original_param->data[i]) {
            param_changed = 1;
            break;
        }
    }
    
    assert(param_changed);
    
    // Check that gradients were reset
    float grad_sum = matrix_sum(grad);
    assert(fabs(grad_sum) < 1e-6);
    
    printf("SGD optimizer: PASSED\n");
    
    // Cleanup
    matrix_free(original_param);
    matrix_free(param);
    matrix_free(grad);
    free(optimizer->params);
    free(optimizer->grads);
    optimizer->free(optimizer);
}

void test_sgd_optimizer_with_momentum() {
    printf("Testing SGD optimizer with momentum...\n");
    
    // Create an SGD optimizer with momentum
    Optimizer* optimizer = sgd_optimizer(0.01f, 0.9f);
    
    // Create a parameter matrix
    Matrix* param = matrix_create(2, 2);
    matrix_fill(param, 1.0f);
    
    // Create a gradient matrix
    Matrix* grad = matrix_create(2, 2);
    matrix_fill(grad, 0.1f);
    
    // Set up optimizer
    optimizer->params = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->params[0] = param;
    optimizer->grads[0] = grad;
    optimizer->param_count = 1;
    
    // Store original parameter values
    Matrix* original_param = matrix_create(param->rows, param->cols);
    matrix_copy(original_param, param);
    
    // Run optimizer update
    optimizer->update(optimizer);
    
    // Check that parameters were updated
    int param_changed = 0;
    for (size_t i = 0; i < param->rows * param->cols; i++) {
        if (param->data[i] != original_param->data[i]) {
            param_changed = 1;
            break;
        }
    }
    
    assert(param_changed);
    
    printf("SGD optimizer with momentum: PASSED\n");
    
    // Cleanup
    matrix_free(original_param);
    matrix_free(param);
    matrix_free(grad);
    free(optimizer->params);
    free(optimizer->grads);
    optimizer->free(optimizer);
}

void test_adam_optimizer() {
    printf("Testing Adam optimizer...\n");
    
    // Create an Adam optimizer
    Optimizer* optimizer = adam_optimizer(0.01f, 0.9f, 0.999f, 1e-8f);
    
    // Create a parameter matrix
    Matrix* param = matrix_create(2, 2);
    matrix_fill(param, 1.0f);
    
    // Create a gradient matrix
    Matrix* grad = matrix_create(2, 2);
    matrix_fill(grad, 0.1f);
    
    // Set up optimizer
    optimizer->params = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->m = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->v = (Matrix**)malloc(sizeof(Matrix*));
    
    optimizer->params[0] = param;
    optimizer->grads[0] = grad;
    optimizer->m[0] = matrix_create(2, 2);
    optimizer->v[0] = matrix_create(2, 2);
    matrix_fill(optimizer->m[0], 0.0f);
    matrix_fill(optimizer->v[0], 0.0f);
    optimizer->param_count = 1;
    
    // Store original parameter values
    Matrix* original_param = matrix_create(param->rows, param->cols);
    matrix_copy(original_param, param);
    
    // Run optimizer update
    optimizer->update(optimizer);
    
    // Check that parameters were updated
    int param_changed = 0;
    for (size_t i = 0; i < param->rows * param->cols; i++) {
        if (param->data[i] != original_param->data[i]) {
            param_changed = 1;
            break;
        }
    }
    
    assert(param_changed);
    
    printf("Adam optimizer: PASSED\n");
    
    // Cleanup
    matrix_free(original_param);
    matrix_free(param);
    matrix_free(grad);
    matrix_free(optimizer->m[0]);
    matrix_free(optimizer->v[0]);
    free(optimizer->params);
    free(optimizer->grads);
    free(optimizer->m);
    free(optimizer->v);
    optimizer->free(optimizer);
}

void test_adam_optimizer_multiple_steps() {
    printf("Testing Adam optimizer multiple steps...\n");
    
    // Create an Adam optimizer
    Optimizer* optimizer = adam_optimizer(0.01f, 0.9f, 0.999f, 1e-8f);
    
    // Create a parameter matrix
    Matrix* param = matrix_create(2, 2);
    matrix_fill(param, 1.0f);
    
    // Create a gradient matrix
    Matrix* grad = matrix_create(2, 2);
    matrix_fill(grad, 0.1f);
    
    // Set up optimizer
    optimizer->params = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->m = (Matrix**)malloc(sizeof(Matrix*));
    optimizer->v = (Matrix**)malloc(sizeof(Matrix*));
    
    optimizer->params[0] = param;
    optimizer->grads[0] = grad;
    optimizer->m[0] = matrix_create(2, 2);
    optimizer->v[0] = matrix_create(2, 2);
    matrix_fill(optimizer->m[0], 0.0f);
    matrix_fill(optimizer->v[0], 0.0f);
    optimizer->param_count = 1;
    
    // Store original parameter values
    Matrix* original_param = matrix_create(param->rows, param->cols);
    matrix_copy(original_param, param);
    
    // Run multiple optimizer updates
    for (int step = 0; step < 5; step++) {
        optimizer->update(optimizer);
    }
    
    // Check that parameters were updated
    int param_changed = 0;
    for (size_t i = 0; i < param->rows * param->cols; i++) {
        if (param->data[i] != original_param->data[i]) {
            param_changed = 1;
            break;
        }
    }
    
    assert(param_changed);
    
    printf("Adam optimizer multiple steps: PASSED\n");
    
    // Cleanup
    matrix_free(original_param);
    matrix_free(param);
    matrix_free(grad);
    matrix_free(optimizer->m[0]);
    matrix_free(optimizer->v[0]);
    free(optimizer->params);
    free(optimizer->grads);
    free(optimizer->m);
    free(optimizer->v);
    optimizer->free(optimizer);
}

void test_optimizer_with_multiple_parameters() {
    printf("Testing optimizer with multiple parameters...\n");
    
    // Create an Adam optimizer
    Optimizer* optimizer = adam_optimizer(0.01f, 0.9f, 0.999f, 1e-8f);
    
    // Create two parameter matrices
    Matrix* param1 = matrix_create(2, 2);
    Matrix* param2 = matrix_create(3, 3);
    matrix_fill(param1, 1.0f);
    matrix_fill(param2, 2.0f);
    
    // Create gradient matrices
    Matrix* grad1 = matrix_create(2, 2);
    Matrix* grad2 = matrix_create(3, 3);
    matrix_fill(grad1, 0.1f);
    matrix_fill(grad2, 0.2f);
    
    // Set up optimizer with two parameters
    optimizer->params = (Matrix**)malloc(2 * sizeof(Matrix*));
    optimizer->grads = (Matrix**)malloc(2 * sizeof(Matrix*));
    optimizer->m = (Matrix**)malloc(2 * sizeof(Matrix*));
    optimizer->v = (Matrix**)malloc(2 * sizeof(Matrix*));
    
    optimizer->params[0] = param1;
    optimizer->params[1] = param2;
    optimizer->grads[0] = grad1;
    optimizer->grads[1] = grad2;
    
    optimizer->m[0] = matrix_create(2, 2);
    optimizer->m[1] = matrix_create(3, 3);
    optimizer->v[0] = matrix_create(2, 2);
    optimizer->v[1] = matrix_create(3, 3);
    
    matrix_fill(optimizer->m[0], 0.0f);
    matrix_fill(optimizer->m[1], 0.0f);
    matrix_fill(optimizer->v[0], 0.0f);
    matrix_fill(optimizer->v[1], 0.0f);
    
    optimizer->param_count = 2;
    
    // Store original parameter values
    Matrix* original_param1 = matrix_create(param1->rows, param1->cols);
    matrix_copy(original_param1, param1);
    Matrix* original_param2 = matrix_create(param2->rows, param2->cols);
    matrix_copy(original_param2, param2);
    
    // Run optimizer update
    optimizer->update(optimizer);
    
    // Check that both parameters were updated
    int param1_changed = 0;
    for (size_t i = 0; i < param1->rows * param1->cols; i++) {
        if (param1->data[i] != original_param1->data[i]) {
            param1_changed = 1;
            break;
        }
    }
    
    int param2_changed = 0;
    for (size_t i = 0; i < param2->rows * param2->cols; i++) {
        if (param2->data[i] != original_param2->data[i]) {
            param2_changed = 1;
            break;
        }
    }
    
    assert(param1_changed);
    assert(param2_changed);
    
    printf("Optimizer with multiple parameters: PASSED\n");
    
    // Cleanup
    matrix_free(original_param1);
    matrix_free(original_param2);
    matrix_free(param1);
    matrix_free(param2);
    matrix_free(grad1);
    matrix_free(grad2);
    matrix_free(optimizer->m[0]);
    matrix_free(optimizer->m[1]);
    matrix_free(optimizer->v[0]);
    matrix_free(optimizer->v[1]);
    free(optimizer->params);
    free(optimizer->grads);
    free(optimizer->m);
    free(optimizer->v);
    optimizer->free(optimizer);
}

int main() {
    printf("Running optimizer tests...\n\n");
    
    test_sgd_optimizer();
    test_sgd_optimizer_with_momentum();
    test_adam_optimizer();
    test_adam_optimizer_multiple_steps();
    test_optimizer_with_multiple_parameters();
    
    printf("\nAll optimizer tests PASSED!\n");
    return 0;
}