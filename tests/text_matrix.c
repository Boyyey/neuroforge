#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/matrix.h"

void test_matrix_operations() {
    printf("Testing basic matrix operations...\n");
    
    // Create matrices
    Matrix* a = matrix_create(2, 2);
    Matrix* b = matrix_create(2, 2);
    Matrix* c = matrix_create(2, 2);
    
    // Fill with test data
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    
    matrix_from_array(a, a_data);
    matrix_from_array(b, b_data);
    
    // Test matrix addition
    matrix_copy(c, a);
    matrix_add(c, b);
    
    assert(fabs(c->data[0] - 6.0f) < 1e-6);  // 1 + 5
    assert(fabs(c->data[1] - 8.0f) < 1e-6);  // 2 + 6
    assert(fabs(c->data[2] - 10.0f) < 1e-6); // 3 + 7
    assert(fabs(c->data[3] - 12.0f) < 1e-6); // 4 + 8
    
    // Test matrix subtraction
    matrix_copy(c, a);
    matrix_subtract(c, b);
    
    assert(fabs(c->data[0] - (-4.0f)) < 1e-6);  // 1 - 5
    assert(fabs(c->data[1] - (-4.0f)) < 1e-6);  // 2 - 6
    assert(fabs(c->data[2] - (-4.0f)) < 1e-6);  // 3 - 7
    assert(fabs(c->data[3] - (-4.0f)) < 1e-6);  // 4 - 8
    
    // Test matrix scaling
    matrix_copy(c, a);
    matrix_scale(c, 2.0f);
    
    assert(fabs(c->data[0] - 2.0f) < 1e-6);  // 1 * 2
    assert(fabs(c->data[1] - 4.0f) < 1e-6);  // 2 * 2
    assert(fabs(c->data[2] - 6.0f) < 1e-6);  // 3 * 2
    assert(fabs(c->data[3] - 8.0f) < 1e-6);  // 4 * 2
    
    printf("Basic matrix operations: PASSED\n");
    
    // Cleanup
    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

void test_matrix_multiplication() {
    printf("Testing matrix multiplication...\n");
    
    // Create matrices for multiplication
    Matrix* a = matrix_create(2, 3);
    Matrix* b = matrix_create(3, 2);
    Matrix* c = matrix_create(2, 2);
    
    // Fill with test data
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    
    matrix_from_array(a, a_data);
    matrix_from_array(b, b_data);
    
    // Perform multiplication: C = A * B
    matrix_multiply(a, b, c);
    
    // Expected result:
    // [1 2 3] * [7  8 ] = [1*7+2*9+3*11  1*8+2*10+3*12]
    // [4 5 6]   [9  10]   [4*7+5*9+6*11  4*8+5*10+6*12]
    //           [11 12]
    // = [7+18+33  8+20+36] = [58  64]
    //   [28+45+66 32+50+72]   [139 154]
    
    assert(fabs(c->data[0] - 58.0f) < 1e-6);
    assert(fabs(c->data[1] - 64.0f) < 1e-6);
    assert(fabs(c->data[2] - 139.0f) < 1e-6);
    assert(fabs(c->data[3] - 154.0f) < 1e-6);
    
    printf("Matrix multiplication: PASSED\n");
    
    // Cleanup
    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

void test_matrix_utility_functions() {
    printf("Testing matrix utility functions...\n");
    
    // Create a test matrix
    Matrix* m = matrix_create(2, 3);
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    matrix_from_array(m, data);
    
    // Test matrix sum
    float sum = matrix_sum(m);
    assert(fabs(sum - 21.0f) < 1e-6);  // 1+2+3+4+5+6 = 21
    
    // Test matrix max
    float max_val = matrix_max(m);
    assert(fabs(max_val - 6.0f) < 1e-6);
    
    // Test matrix min
    float min_val = matrix_min(m);
    assert(fabs(min_val - 1.0f) < 1e-6);
    
    // Test matrix fill
    matrix_fill(m, 10.0f);
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        assert(fabs(m->data[i] - 10.0f) < 1e-6);
    }
    
    printf("Matrix utility functions: PASSED\n");
    
    // Cleanup
    matrix_free(m);
}

void test_matrix_views() {
    printf("Testing matrix views...\n");
    
    // Create a parent matrix
    Matrix* parent = matrix_create(3, 3);
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    matrix_from_array(parent, data);
    
    // Create a view of the top-left 2x2 submatrix
    Matrix* view = matrix_view(parent, 0, 0, 2, 2);
    
    // Check view contents
    assert(view->data[0 * view->stride + 0] == 1.0f);  // [0,0]
    assert(view->data[0 * view->stride + 1] == 2.0f);  // [0,1]
    assert(view->data[1 * view->stride + 0] == 4.0f);  // [1,0]
    assert(view->data[1 * view->stride + 1] == 5.0f);  // [1,1]
    
    // Modify view (should affect parent)
    view->data[0] = 100.0f;
    assert(parent->data[0] == 100.0f);
    
    printf("Matrix views: PASSED\n");
    
    // Cleanup
    matrix_free(view);
    matrix_free(parent);
}

int main() {
    printf("Running matrix tests...\n\n");
    
    test_matrix_operations();
    test_matrix_multiplication();
    test_matrix_utility_functions();
    test_matrix_views();
    
    printf("\nAll matrix tests PASSED!\n");
    return 0;
}