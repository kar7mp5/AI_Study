#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 8
#define HIDDEN_LAYER 3
#define POOL_SIZE 2
#define OUTPUT_SIZE 4

#define EPSILON 1e-7

#define relu(x) ((x) > 0 ? (x) : 0)

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
 * @brief Initialize a 1D array randomly.
 * 
 * @param data Pointer to the 1D array holding the data to be initialized.
 * @param size The number of array size.
 */
void initialize_random(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2 - 1; // 값 범위: -1 ~ 1
    }
}

/**
 * @brief Applies a convolution operation on the input matrix using the given filter.
 *
 * @param input The input matrix with dimensions.
 * @param filter The filter matrix with dimensions.
 * @param output The output matrix to store the convolution result. It has dimensions.
 */
void convolution(float input[INPUT_SIZE][INPUT_SIZE],
                 float filter[HIDDEN_LAYER][HIDDEN_LAYER],
                 float output[INPUT_SIZE - HIDDEN_LAYER + 1]
                             [INPUT_SIZE - HIDDEN_LAYER + 1]) {
    for (int rows = 0; rows < INPUT_SIZE - HIDDEN_LAYER + 1; ++rows) {
        for (int cols = 0; cols < INPUT_SIZE - HIDDEN_LAYER + 1; ++cols) {
            float sum = 0;
            for (int fi = 0; fi < HIDDEN_LAYER; ++fi) {
                for (int fj = 0; fj < HIDDEN_LAYER; ++fj) {
                    sum += input[rows + fi][cols + fj] * filter[fi][fj];
                }
            }
            output[rows][cols] = relu(sum);
        }
    }
}

/**
 * @brief Performs max pooling operation on the input data.
 *
 * @param input The input matrix that Convolution matrix.
 */
void max_pooling(float input[INPUT_SIZE - HIDDEN_LAYER + 1][INPUT_SIZE - HIDDEN_LAYER + 1], 
                 float output[(INPUT_SIZE - HIDDEN_LAYER + 1)/POOL_SIZE]
                             [(INPUT_SIZE - HIDDEN_LAYER + 1)/POOL_SIZE]) {
    for (int rows = 0; rows < INPUT_SIZE - HIDDEN_LAYER + 1; rows += POOL_SIZE) {
        for (int cols = 0; cols < INPUT_SIZE - HIDDEN_LAYER + 1; cols += POOL_SIZE) {
            float max = -INFINITY;
            for (int pi = 0; pi < POOL_SIZE; ++pi) {
                for (int pj = 0; pj < POOL_SIZE; ++pj) {
                    if (input[rows + pi][cols + pj] > max) {
                        max = input[rows + pi][cols + pj];
                    }
                }
            }
            output[rows / POOL_SIZE][cols / POOL_SIZE] = max;
        }
    }
}

/**
 * @brief Applies Fully Connected Layer.
 *
 * @param input The input matrix.
 * @param weights The matrix of weights.
 * @param bias The matrix of bias.
 * @param output The output matrix.
 */
void fully_connected(float input[], float weights[OUTPUT_SIZE][INPUT_SIZE], 
                     float bias[OUTPUT_SIZE], float output[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = bias[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            output[i] = input[j] * weights[i][j];
        }
    }
}

/**
 * @brief Approximates the exponential function e^x using Taylor series.
 *
 * @param x The exponent value.
 * @return The computed value of e^x.
 */
double my_exp(double x) {
    double result = 1.0;
    double term = 1.0;
    int n = 1; 

    while (term > EPSILON || term < -EPSILON) {
        term *= x / n;
        result += term;
        n++;
    }

    return result;
}

/**
 * @brief Computes the softmax for an array of values.
 *
 * @param input The input array of size n.
 * @param output The output array to store the softmax probabilities (size n).
 * @param n The size of the input/output arrays.
 */
void softmax(float *input, float *output, int n) {
    float max_val = input[0];
    float sum = 0.0;

    // Find the maximum value in the input to stabilize the computation.
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Compute the exponentials of the adjusted values and their sum.
    for (int i = 0; i < n; i++) {
        output[i] = my_exp(input[i] - max_val); // Subtract max_val for numerical stability.
        sum += output[i];
    }

    // Normalize the exponentials to compute the softmax probabilities.
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// Backpropagation function
void backward_propagation(float input_layer[INPUT_SIZE][INPUT_SIZE], float hidden_layer[HIDDEN_LAYER][HIDDEN_LAYER], 
                          float conv_output[INPUT_SIZE - HIDDEN_LAYER + 1][INPUT_SIZE - HIDDEN_LAYER + 1], 
                          float pooled_output[(INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE][(INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE], 
                          float fc_weights[OUTPUT_SIZE][INPUT_SIZE], float fc_bias[OUTPUT_SIZE], 
                          float output_layer[OUTPUT_SIZE], float expected_output[OUTPUT_SIZE]) {
    
    // Step 1: Calculate the softmax loss and gradient (Cross-entropy loss)
    float softmax_output[OUTPUT_SIZE];
    softmax(output_layer, softmax_output, OUTPUT_SIZE);

    // Gradients for output layer (dL/dy)
    float output_grad[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_grad[i] = softmax_output[i] - expected_output[i];
    }

    // Step 2: Backpropagate the gradients to the fully connected layer
    float fc_weights_grad[OUTPUT_SIZE][INPUT_SIZE];
    float fc_bias_grad[OUTPUT_SIZE];
    
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        fc_bias_grad[i] = output_grad[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            fc_weights_grad[i][j] = output_grad[i] * pooled_output[i][j];
        }
    }

    // Step 3: Update the fully connected layer weights and biases
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        fc_bias[i] -= LEARNING_RATE * fc_bias_grad[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            fc_weights[i][j] -= LEARNING_RATE * fc_weights_grad[i][j];
        }
    }

    // Step 4: Backpropagate the gradients to the pooling layer and convolutional layer
    // Flatten the pooled output
    float pooled_grad[(INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE][(INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE];
    for (int i = 0; i < (INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE; ++i) {
        for (int j = 0; j < (INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE; ++j) {
            pooled_grad[i][j] = 0;  // Initialize gradient array
        }
    }

    // Max pooling layer gradient
    for (int rows = 0; rows < INPUT_SIZE - HIDDEN_LAYER + 1; rows += POOL_SIZE) {
        for (int cols = 0; cols < INPUT_SIZE - HIDDEN_LAYER + 1; cols += POOL_SIZE) {
            float max = pooled_output[rows / POOL_SIZE][cols / POOL_SIZE];
            for (int pi = 0; pi < POOL_SIZE; ++pi) {
                for (int pj = 0; pj < POOL_SIZE; ++pj) {
                    if (conv_output[rows + pi][cols + pj] == max) {
                        pooled_grad[rows / POOL_SIZE][cols / POOL_SIZE] += output_grad[rows + pi][cols + pj];
                    }
                }
            }
        }
    }

    // Backpropagate the gradients to the convolutional layer filters
    float filter_grad[HIDDEN_LAYER][HIDDEN_LAYER];
    for (int fi = 0; fi < HIDDEN_LAYER; ++fi) {
        for (int fj = 0; fj < HIDDEN_LAYER; ++fj) {
            filter_grad[fi][fj] = 0;
            for (int i = 0; i < INPUT_SIZE - HIDDEN_LAYER + 1; ++i) {
                for (int j = 0; j < INPUT_SIZE - HIDDEN_LAYER + 1; ++j) {
                    filter_grad[fi][fj] += conv_output[i][j] * pooled_grad[i][j];
                }
            }
        }
    }

    // Update the convolutional filters using gradient descent
    for (int fi = 0; fi < HIDDEN_LAYER; ++fi) {
        for (int fj = 0; fj < HIDDEN_LAYER; ++fj) {
            hidden_layer[fi][fj] -= LEARNING_RATE * filter_grad[fi][fj];
        }
    }
}

void backward_propagation(output_)

int main() {
    srand(time(0));
    // srand(0);

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
    max_pooling(conv_output, pooled_output);

    // Backward
    backward_propagation(input_layer, hidden_layer, conv_output, pooled_output, fc_weights, fc_bias, output_layer, expected_output);

    printf("Convolution Layer Output\n");
    print_layer((float *)conv_output, INPUT_SIZE - HIDDEN_LAYER + 1, INPUT_SIZE - HIDDEN_LAYER + 1);
    printf("Pooled Output\n");
    print_layer((float *)pooled_output, (INPUT_SIZE - HIDDEN_LAYER + 1)/POOL_SIZE, (INPUT_SIZE - HIDDEN_LAYER + 1)/POOL_SIZE);

    // Flatten for Fully Connected Layer
    float flattened[(INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE *
                    (INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE];
    int idx = 0;   
    for (int i = 0; i < (INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE; ++i) {
        for (int j = 0; j < (INPUT_SIZE - HIDDEN_LAYER + 1) / POOL_SIZE; ++j) {
            flattened[idx++] = pooled_output[i][j];
        }
    }

    float fc_output[OUTPUT_SIZE];

    // Fully Connected Layer
    fully_connected(flattened, fc_weights, fc_bias, fc_output);

    printf("Output Layer\n");
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        printf("%+6.2f", fc_output[i]);
    }
    printf("\n\n");

    // Calculate result
    float output[OUTPUT_SIZE];
    softmax(fc_output, output, OUTPUT_SIZE);
    printf("Output(Softmax)\n");
     for (int i = 0; i < OUTPUT_SIZE; ++i) {
        printf("%+6.2f", output[i]);
    }
    printf("\n\n");

    int index = 0;
    float max = -INFINITY;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        if (output[i] > max) {
            index = i;
            max = output[i];
        }
    }
    printf("Result: %d\n", index);

    return 0;
}