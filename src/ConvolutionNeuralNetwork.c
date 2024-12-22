// forward_convolutionNeuralNetwork.c

#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPSILON 1e-7

// Define ReLU functions
#define relu(x) ((x) > 0 ? (x) : 0)
#define relu_derivative(x) ((x) > 0 ? 1 : 0)

/**
 * @brief Applies Mean Squared Error operation.
 *
 * @param output The output matrix of past data.
 * @param target The target matrix of post data.
 * @return The computed Mean Squared Error loss as a double value.
 */
double mse_loss(Matrix *output, Matrix *target) {
    double loss = 0;
    for (int i = 0; i < output->rows; ++i) {
        for (int j = 0; j < output->cols; ++j) {
            double diff = output->data[i][j] - target->data[i][j];
            loss += diff * diff;
        }
    }
    return loss / (output->rows * output->cols);
}

/**
 * @brief Computes the derivative of the Mean Squared Error (MSE) loss.
 *
 * dL/d(output) = 2 * (output - target) / N
 *
 * @param output The output matrix of past data, which contains predictions or calculated values.
 * @param target The target matrix of post data, which contains ground truth values.
 * @param d_output The matrix to store the computed derivatives, same size as the output matrix.
 */
void mse_loss_derivative(Matrix *output, Matrix *target, Matrix *d_output) {
    for (int i = 0; i < output->rows; ++i) {
        for (int j = 0; j < output->cols; ++j) {
            d_output->data[i][j] = 2 * (output->data[i][j] - target->data[i][j]) / (output->rows * output->cols);
        }
    }
}

/**
 * @brief Applies a forward_convolution operation.
 *
 * @param input The input matrix with dimentsions.
 * @param filter The filter matrix with dimentions.
 * @param output The output matrix to store the forward_convolution result.
 */
void forward_convolution(Matrix *input, Matrix *filter, Matrix *output) {
    // Validate the matrix format
    if (input->rows != input->cols) {
        printf("Invaild input matrix format!");
        return;
    }
    if (filter->rows != filter->cols) {
        printf("Invaild filter matrix format!");
        return;
    }

    // Define matrices size
    int input_size = input->rows;
    int filter_size = filter->rows;
    int output_size = input_size - filter_size + 1;

    // Create intermediate matrices
    Matrix input_matrix = create_matrix(output_size * output_size, filter_size * filter_size);
    Matrix filter_matrix = create_matrix(filter_size * filter_size, 1);

    // Reallocate the input matrix to flatten matrix
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            for (int fi = 0; fi < filter_size; ++fi) {
                for (int fj = 0; fj < filter_size; ++fj) {
                    input_matrix.data[i * output_size + j][fi * filter_size + fj] = input->data[i + fi][j + fj];
                }
            }
        }
    }

    // Reallocate the filter matrix to flatten matrix
    for (int fi = 0; fi < filter_size; ++fi) {
        for (int fj = 0; fj < filter_size; ++fj) {
            filter_matrix.data[fi * filter_size + fj][0] = filter->data[fi][fj];
        }
    }

    // Perform matrix multiplication
    Matrix output_matrix = create_matrix(output_size * output_size, 1);
    matmul(&input_matrix, &filter_matrix, &output_matrix);

    // Reshape output_matrix back into output and apply ReLU
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            output->data[i][j] = relu(output_matrix.data[i * output_size + j][0]);
        }
    }

    // Free intermediate metrix
    free_matrix(&input_matrix);
    free_matrix(&filter_matrix);
    free_matrix(&output_matrix);
}

/**
 * @brief Updates a filter matrix using the gradient descent algorithm.
 *
 * @param filter The filter matrix to be updated, typically representing model parameters.
 * @param d_filter The gradient matrix containing the partial derivatives of the loss
 *                 function with respect to the filter matrix elements.
 * @param learning_rate A constant value controlling the step size of the gradient descent update.
 */
void update_filter(Matrix *filter, Matrix *d_filter, const double learning_rate) {
    for (int i = 0; i < filter->rows; ++i) {
        for (int j = 0; j < filter->cols; ++j) {
            filter->data[i][j] -= learning_rate * d_filter->data[i][j];
        }
    }
}

/**
 * @brief Performs the backward pass for convolution, computing gradients with respect to inputs and filters.
 *
 * @param d_output The gradient matrix of the output, representing the loss gradient with respect to the output.
 * @param filter The filter matrix used during the forward pass.
 * @param d_input The gradient matrix of the input to be computed and updated.
 * @param d_filter The gradient matrix of the filter to be computed and updated.
 */
void backward_convolution(Matrix *d_output, Matrix *filter, Matrix *d_input, Matrix *d_filter) {
    int filter_size = filter->rows;
    int output_size = d_output->rows;

    // Initialize d_filter
    for (int i = 0; i < d_filter->rows; ++i) {
        for (int j = 0; j < d_filter->cols; ++j) {
            d_filter->data[i][j] = 0;
        }
    }

    // Compute d_filter using matmul (flattened_input is not needed)
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            double grad = d_output->data[i][j];
            for (int fi = 0; fi < filter_size; ++fi) {
                for (int fj = 0; fj < filter_size; ++fj) {
                    d_filter->data[fi][fj] += grad * d_input->data[i + fi][j + fj];
                }
            }
        }
    }

    // Compute d_input (no need for flattened_input)
    for (int i = 0; i < d_input->rows; ++i) {
        for (int j = 0; j < d_input->cols; ++j) {
            d_input->data[i][j] = 0; // Reset d_input to zero before accumulation
        }
    }

    // Backpropagate gradient to input using filter
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            double grad = d_output->data[i][j];
            for (int fi = 0; fi < filter_size; ++fi) {
                for (int fj = 0; fj < filter_size; ++fj) {
                    d_input->data[i + fi][j + fj] += grad * filter->data[fi][fj];
                }
            }
        }
    }
}

/**
 * @brief Trains a filter using convolutional forward and backward passes with gradient descent.
 *
 * @param input The input matrix containing training data.
 * @param target The target matrix containing ground truth data.
 * @param filter The filter matrix to be trained (model parameters).
 * @param epochs The number of training iterations.
 * @param learning_rate The learning rate used for gradient descent updates.
 */
void train(Matrix *input, Matrix *target, Matrix *filter, int epochs, double learning_rate) {
    int input_size = input->rows;
    int filter_size = filter->rows;
    int output_size = input_size - filter_size + 1;

    // Create necessary matrices
    Matrix output = create_matrix(output_size, output_size);
    Matrix d_output = create_matrix(output_size, output_size);
    Matrix d_input = create_matrix(input_size, input_size);
    Matrix d_filter = create_matrix(filter_size, filter_size);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // printf("Forward\n");
        // Forward pass
        forward_convolution(input, filter, &output);

        // Compute loss
        double loss = mse_loss(&output, target);

        // Backward pass
        // printf("Loss\n");
        mse_loss_derivative(&output, target, &d_output);
        // printf("Backward\n");
        backward_convolution(&d_output, filter, &d_input, &d_filter);

        // Update filter
        update_filter(filter, &d_filter, learning_rate);

        // Print loss for this epoch
        printf("Epoch %d, Loss: %.4f\n", epoch + 1, loss);
    }

    // Free allocated memory
    free_matrix(&output);
    free_matrix(&d_output);
    free_matrix(&d_input);
    free_matrix(&d_filter);
}

int main() {
    srand(0);

    int input_size = 5;
    int filter_size = 3;
    int output_size = input_size - filter_size + 1;

    int epochs = 50;
    double learning_rate = 0.0001;

    // Create input matrix
    Matrix input = create_matrix(input_size, input_size);

    // Initialize random seed
    srand(time(NULL));

    // Fill input matrix with random values
    double input_data[input_size * input_size];
    for (int i = 0; i < input_size * input_size; i++) {
        input_data[i] = (double)(rand() % 100); // Generates random values between 0 and 99
    }
    initialize_matrix(&input, input_data);

    // Create target matrix
    Matrix target = create_matrix(output_size, output_size);
    double target_data[] = {10, 20, 30, 20, 10, 5, 15, 25, 35};
    initialize_matrix(&target, target_data);

    // Create filter matrix
    Matrix filter = create_matrix(filter_size, filter_size);
    double filter_data[] = {0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5};
    initialize_matrix(&filter, filter_data);

    // Train the CNN
    train(&input, &target, &filter, epochs, learning_rate);

    // Print the final filter
    printf("\nTrained Filter:\n");
    for (int i = 0; i < filter.rows; ++i) {
        for (int j = 0; j < filter.cols; ++j) {
            printf("%8.4f ", filter.data[i][j]);
        }
        printf("\n");
    }

    // Clean up
    free_matrix(&input);
    free_matrix(&target);
    free_matrix(&filter);

    return 0;
}