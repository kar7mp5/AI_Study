// ConvolutionNeuralNetwork.c

#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
// #include <stdlib.h>

#define INPUT_SIZE 8
#define FILTER_SIZE 3
#define POOL_SIZE 4
#define OUTPUT_SIZE 4

#define EPSILON 1e-7

#define relu(x) ((x) > 0 ? (x) : 0)

int main() {
    int r1 = 2, r2 = 3, c2 = 2;

    Matrix m1 = create_matrix(r1, r2);
    Matrix m2 = create_matrix(r2, c2);
    Matrix output = create_matrix(r1, c2);

    float m1_data[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    float m2_data[3][2] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};

    initialize_matrix(&m1, &m1_data);
    initialize_matrix(&m2, &m2_data);

    print_matrix(&m1);

    matmul(&m1, &m2, &output);

    printf("Resultant Matrix:\n");
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            printf("%.2f ", output.data[i][j]);
        }
        printf("\n");
    }

    free_matrix(&m1);
    free_matrix(&m2);
    free_matrix(&output);

    return 0;
}
