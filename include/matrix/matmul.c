// matmul.c

#include "matrix/matmul.h"
#include <stdlib.h>

// Functioin to perform matrix multiplication
void matmul(Matrix *m1, Matrix *m2, Matrix *output) {
    // Verify if matrix multiplication is possible
    if (m1->cols != m2->rows) {
        printf("Matrix dimensions do not match for multiplication.\n");
        return;
    }

    for (int i = 0; i < m1->rows; ++i) {
        for (int j = 0; j < m2->cols; ++j) {
            output->data[i][j] = 0;
            for (int k = 0; k < m1->cols; ++k) {
                output->data[i][j] += m1->data[i][k] * m2->data[k][j];
            }
        }
    }
}

// Function to create a matrix with given rows and columns
Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;

    // Allocate memory for the matrix
    m.data = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; ++i) {
        m.data[i] = (float *)malloc(cols * sizeof(float));
    }

    return m;
}

// Function to initialize a dynamically allocated matrix
void initialize_matrix(Matrix *matrix, const float *data) {
    int idx = 0;
    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = 0; j < matrix->cols; ++j) {
            matrix->data[i][j] = data[idx++];
        }
    }
}

// Function to free the memory of a matrix
void free_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; ++i) {
        free(m->data[i]);
    }
    free(m->data);
}

// Print a matrix
void print_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = 0; j < matrix->cols; ++j) {
            printf("%+8.2f", matrix->data[i][j]);
        }
        printf("\n");
    }
}