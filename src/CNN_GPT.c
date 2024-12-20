#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 28
#define FILTER_SIZE 3
#define POOL_SIZE 2
#define OUTPUT_SIZE 10

// Activation Function
float relu(float x) { return x > 0 ? x : 0; }

// Convolution
void convolution(
    float input[INPUT_SIZE][INPUT_SIZE], float filter[FILTER_SIZE][FILTER_SIZE],
    float output[INPUT_SIZE - FILTER_SIZE + 1][INPUT_SIZE - FILTER_SIZE + 1]) {
    for (int i = 0; i < INPUT_SIZE - FILTER_SIZE + 1; i++) {
        for (int j = 0; j < INPUT_SIZE - FILTER_SIZE + 1; j++) {
            float sum = 0;
            for (int fi = 0; fi < FILTER_SIZE; fi++) {
                for (int fj = 0; fj < FILTER_SIZE; fj++) {
                    sum += input[i + fi][j + fj] * filter[fi][fj];
                }
            }
            output[i][j] = relu(sum);
        }
    }
}

// Max Pooling
void max_pooling(
    float input[INPUT_SIZE - FILTER_SIZE + 1][INPUT_SIZE - FILTER_SIZE + 1],
    float output[(INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE]
                [(INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE]) {
    for (int i = 0; i < (INPUT_SIZE - FILTER_SIZE + 1); i += POOL_SIZE) {
        for (int j = 0; j < (INPUT_SIZE - FILTER_SIZE + 1); j += POOL_SIZE) {
            float max = -INFINITY;
            for (int pi = 0; pi < POOL_SIZE; pi++) {
                for (int pj = 0; pj < POOL_SIZE; pj++) {
                    if (input[i + pi][j + pj] > max) {
                        max = input[i + pi][j + pj];
                    }
                }
            }
            output[i / POOL_SIZE][j / POOL_SIZE] = max;
        }
    }
}

// Fully Connected Layer
void fully_connected(float input[], float weights[OUTPUT_SIZE][INPUT_SIZE],
                     float bias[OUTPUT_SIZE], float output[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = bias[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            output[i] += input[j] * weights[i][j];
        }
    }
}

// Initialize random data
void initialize_random(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2 - 1; // 값 범위: -1 ~ 1
    }
}

// Main
int main() {
    srand(time(0));

    // Input Data (28x28 이미지)
    float input[INPUT_SIZE][INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[i][j] = ((float)rand() / RAND_MAX); // 0~1 사이의 값
        }
    }

    // Filter (3x3)
    float filter[FILTER_SIZE][FILTER_SIZE];
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            filter[i][j] = ((float)rand() / RAND_MAX) * 2 - 1; // -1 ~ 1
        }
    }

    // Convolution Layer Output
    float conv_output[INPUT_SIZE - FILTER_SIZE + 1]
                     [INPUT_SIZE - FILTER_SIZE + 1];

    // Max Pooling Output
    float pooled_output[(INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE]
                       [(INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE];

    // Fully Connected Layer Weights and Biases
    float fc_weights[OUTPUT_SIZE][INPUT_SIZE];
    float fc_bias[OUTPUT_SIZE];
    initialize_random((float *)fc_weights, OUTPUT_SIZE * INPUT_SIZE);
    initialize_random(fc_bias, OUTPUT_SIZE);

    // Fully Connected Output
    float fc_output[OUTPUT_SIZE];

    // Forward Pass
    convolution(input, filter, conv_output);
    max_pooling(conv_output, pooled_output);

    // Flatten for Fully Connected Layer
    float flattened[(INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE *
                    (INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE];
    int idx = 0;
    for (int i = 0; i < (INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE; i++) {
        for (int j = 0; j < (INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE; j++) {
            flattened[idx++] = pooled_output[i][j];
        }
    }

    fully_connected(flattened, fc_weights, fc_bias, fc_output);

    // Output
    printf("Final Outputs:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Class %d: %f\n", i, fc_output[i]);
    }

    return 0;
}
