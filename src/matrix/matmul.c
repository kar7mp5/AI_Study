// matmul.c

#include "matrix/matmul.h"
#include <stdio.h>
#include <stdlib.h>

// Function to perform matrix multiplication
void matmul(Matrix *m1, Matrix *m2, Matrix *output) {
    // Verify if matrix multiplication is possible
    if (m1->cols != m2->rows) {
        printf("Matrix dimensions do not match for multiplication: A(%d, %d) * B(%d, %d)\n", m1->rows, m1->cols,
               m2->rows, m2->cols);
        return;
    }

    // Check if output matrix has correct size
    if (output->rows != m1->rows || output->cols != m2->cols) {
        printf("Output matrix size does not match multiplication result size.\n");
        return;
    }

    // Perform matrix multiplication
    for (int i = 0; i < m1->rows; ++i) {
        for (int j = 0; j < m2->cols; ++j) {
            output->data[i * output->cols + j] = 0; // Correct 1D indexing
            for (int k = 0; k < m1->cols; ++k) {
                output->data[i * output->cols + j] += m1->data[i * m1->cols + k] * m2->data[k * m2->cols + j];
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
    m.data = (double *)malloc(rows * cols * sizeof(double)); // Allocate a single contiguous block of memory
    if (!m.data) {
        printf("Failed to allocate memory for matrix data.\n");
        exit(EXIT_FAILURE);
    }

    return m;
}

// Function to initialize a dynamically allocated matrix
void initialize_matrix(Matrix *matrix, const double *data) {
    int idx = 0;
    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = 0; j < matrix->cols; ++j) {
            matrix->data[i * matrix->cols + j] = data[idx++]; // Correct 1D indexing
        }
    }
}

// Function to free the memory of a matrix
void free_matrix(Matrix *m) {
    if (!m || !m->data)
        return;

    free(m->data);  // Free the single contiguous block of memory
    m->data = NULL; // Prevent dangling pointer
}

// Transposes a given matrix
Matrix transpose_matrix(Matrix *m) {
    Matrix t = create_matrix(m->cols, m->rows);

    for (int i = 0; i < m->rows; ++i) {
        for (int j = 0; j < m->cols; ++j) {
            t.data[j * t.cols + i] = m->data[i * m->cols + j]; // Correct 1D indexing
        }
    }
    return t;
}

// Print a matrix
void print_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = 0; j < matrix->cols; ++j) {
            printf("%+8.2f ", matrix->data[i * matrix->cols + j]); // Correct 1D indexing
        }
        printf("\n");
    }
}
