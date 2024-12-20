#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 5
#define HIDDEN_LAYER 3
#define POOL_SIZE 2
#define OUTPUT_SIZE 10

// Activation Function
float relu(const float x) { return x > 0 ? x : 0; }

/**
 * @brief Prints the elements of a 1D array in a grid format.
 *
 * @param data Pointer to the 1D array holding the data to be printed.
 * @param rows The number of rows to print.
 * @param cols The number of columns to print.
 */
void print_layer(float *data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%+6.2f ", data[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Applies a convolution operation on the input matrix using the given
 * filter.
 *
 * @param input The input matrix with dimensions [INPUT_SIZE][INPUT_SIZE].
 * @param filter The filter matrix with dimensions [FILTER_SIZE][FILTER_SIZE].
 * @param output The output matrix to store the convolution result.
 *               It has dimensions [(INPUT_SIZE - FILTER_SIZE + 1)][(INPUT_SIZE
 * - FILTER_SIZE + 1)].
 */
void convolution(float input[INPUT_SIZE][INPUT_SIZE],
                 float filter[HIDDEN_LAYER][HIDDEN_LAYER],
                 float output[INPUT_SIZE - HIDDEN_LAYER + 1]
                             [INPUT_SIZE - HIDDEN_LAYER + 1]) {}

void initialize_random(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2 - 1; // 값 범위: -1 ~ 1
    }
}

int main() {
    srand(time(0));

    // input_layer Data (28x28)
    printf("Initialize Input Layer\n");
    float input_layer[INPUT_SIZE][INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            input_layer[i][j] = ((float)rand() / RAND_MAX);
        }
    }
    print_layer((float *)input_layer, INPUT_SIZE, INPUT_SIZE);

    // hidden_layer (3x3)
    printf("Initialize Hidden Layer\n");
    float hidden_layer[HIDDEN_LAYER][HIDDEN_LAYER];
    for (int i = 0; i < HIDDEN_LAYER; ++i) {
        for (int j = 0; j < HIDDEN_LAYER; ++j) {
            hidden_layer[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }
    print_layer((float *)hidden_layer, HIDDEN_LAYER, HIDDEN_LAYER);

    // Convolution Layer Output
    float conv_output[INPUT_SIZE - HIDDEN_LAYER + 1]
                     [INPUT_SIZE - HIDDEN_LAYER + 1];

    // Max Pooling Output
    float pooled_output[(INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE]
                       [(INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE];

    // Fully Connected Layer Weights and Biases
    float fc_weights[OUTPUT_SIZE][INPUT_SIZE];
    float fc_bias[OUTPUT_SIZE];
    initialize_random((float *)fc_weights, OUTPUT_SIZE * INPUT_SIZE);
    initialize_random(fc_bias, OUTPUT_SIZE);

    printf("Initialize Fully Connected Layer Weights\n");
    print_layer((float *)fc_weights, OUTPUT_SIZE, INPUT_SIZE);
    printf("Biases:\n");
    print_layer(fc_bias, 1, OUTPUT_SIZE);

    // Output
    float output_layer[OUTPUT_SIZE];

    // Forward
    convolution(input_layer, hidden_layer, conv_output);

    return 0;
}