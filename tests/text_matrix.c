#include <stdio.h>
#include <assert.h>
#include "../src/matrix.h"

void test_matrix_creation() {
    printf("Testing matrix creation...\n");
    
    Matrix* m = matrix_create(3, 4);
    assert(m != NULL);
    assert(m->rows == 3);
    assert(m->cols == 4);
    assert(m->stride == 4);
    assert(m->is_view == 0);
    
    matrix_free(m);
    printf("Matrix creation test passed!\n");
}

void test_matrix_operations() {
    printf("Testing matrix operations...\n");
    
    Matrix* a = matrix_create(2, 3);
    Matrix* b = matrix_create(2, 3);
    Matrix* c = matrix_create(2, 3);
    
    // Fill matrices with test values
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {7, 8, 9, 10, 11, 12};
    
    memcpy(a->data, a_data, sizeof(a_data));
    memcpy(b->data, b_data, sizeof(b_data));
    
    // Test addition
    matrix_copy(c, a);
    matrix_add(c, b);
    
    assert(c->data[0] == 8);
    assert(c->data[1] == 10);
    assert(c->data[2] == 12);
    assert(c->data[3] == 14);
    assert(c->data[4] == 16);
    assert(c->data[5] == 18);
    
    // Test subtraction
    matrix_copy(c, b);
    matrix_subtract(c, a);
    
    assert(c->data[0] == 6);
    assert(c->data[1] == 6);
    assert(c->data[2] == 6);
    assert(c->data[3] == 6);
    assert(c->data[4] == 6);
    assert(c->data[5] == 6);
    
    // Test scaling
    matrix_copy(c, a);
    matrix_scale(c, 2.0f);
    
    assert(c->data[0] == 2);
    assert(c->data[1] == 4);
    assert(c->data[2] == 6);
    assert(c->data[3] == 8);
    assert(c->data[4] == 10);
    assert(c->data[5] == 12);
    
    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
    printf("Matrix operations test passed!\n");
}

void test_matrix_multiplication() {
    printf("Testing matrix multiplication...\n");
    
    Matrix* a = matrix_create(2, 3);
    Matrix* b = matrix_create(3, 2);
    Matrix* c = matrix_create(2, 2);
    
    // Fill matrices with test values
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {7, 8, 9, 10, 11, 12};
    
    memcpy(a->data, a_data, sizeof(a_data));
    memcpy(b->data, b_data, sizeof(b_data));
    
    // Test multiplication
    matrix_multiply(a, b, c);
    
    assert(c->data[0] == 58);   // 1*7 + 2*9 + 3*11
    assert(c->data[1] == 64);   // 1*8 + 2*10 + 3*12
    assert(c->data[2] == 139);  // 4*7 + 5*9 + 6*11
    assert(c->data[3] == 154);  // 4*8 + 5*10 + 6*12
    
    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
    printf("Matrix multiplication test passed!\n");
}

int main() {
    printf("Running matrix tests...\n");
    
    test_matrix_creation();
    test_matrix_operations();
    test_matrix_multiplication();
    
    printf("All matrix tests passed!\n");
    return 0;
}