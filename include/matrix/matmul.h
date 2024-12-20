// matmul.h

#ifndef MATH_MATMUL_H
#define MATH_MATMUL_H

#include <stdio.h>

// Define a struct to represent a matrix
typedef struct {
    int rows;     // Number of rows
    int cols;     // Number of columns
    float **data; // Pointer to the matrix data
} Matrix;

/**
 * @brief Performs matrix multiplication.
 * Multiplies a matrix of size r1 x r2 with another matrix of size r2 x c2.
 *
 * @param m1 The first matrix to be multiplied.
 * @param m2 The second matrix to be multiplied.
 * @param output The output matrix to store the result.
 */
void matmul(Matrix *m1, Matrix *m2, Matrix *output);

/**
 * @brief Function to create a matrix with given rows and columns.
 * Allocates memory for the matrix.
 *
 * @param rows Number of rows
 * @param cols Number of cols
 * @return A matrix struct with allocated memory
 */
Matrix create_matrix(int rows, int cols);

/**
 * @brief Function to initialize a dynamically allocated matrix.
 *
 */
void initialize_matrix(Matrix *matrix, const float *data);

/**
 * @brief Function to free the memory of a matrix.
 *
 * @param m The matrix to free
 */
void free_matrix(Matrix *m);

#endif // MATMUL_H
