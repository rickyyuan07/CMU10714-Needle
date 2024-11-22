#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using namespace std;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
// Allocate memory for logits and softmax gradients
    float *logits = new float[batch * k];      // Logits (batch size x num_classes)
    float *probs = new float[batch * k];       // Softmax probabilities (batch size x num_classes)
    float *grad_theta = new float[n * k];      // Gradient for theta (num_features x num_classes)

    // Loop over the dataset in mini-batches
    for (int start = 0; start < m; start += batch) {
        int current_batch_size = std::min(batch, m - start);  // Handle last batch if smaller

        // Step 1: Compute logits for current batch (logits = X_batch * theta)
        memset(logits, 0, sizeof(float) * current_batch_size * k);  // Clear logits
        for (int i = 0; i < current_batch_size; ++i) {
            for (int j = 0; j < k; ++j) {
                for (int p = 0; p < n; ++p) {
                    logits[i * k + j] += X[(start + i) * n + p] * theta[p * k + j];
                }
            }
        }

        // Step 2: Apply softmax to logits to get probabilities
        for (int i = 0; i < current_batch_size; ++i) {
            // Compute softmax probabilities
            float sum_exp = 0.0;
            for (int j = 0; j < k; ++j) {
                probs[i * k + j] = exp(logits[i * k + j]);
                sum_exp += probs[i * k + j];
            }
            for (int j = 0; j < k; ++j) {
                probs[i * k + j] /= sum_exp;
            }
        }

        // Step 3: Compute gradient for theta
        memset(grad_theta, 0, sizeof(float) * n * k);  // Clear gradient
        for (int i = 0; i < current_batch_size; ++i) {
            // Update gradients based on softmax output and true labels
            for (int j = 0; j < k; ++j) {
                float error = probs[i * k + j] - (j == y[start + i] ? 1.0f : 0.0f);  // Softmax error
                for (int p = 0; p < n; ++p) {
                    grad_theta[p * k + j] += X[(start + i) * n + p] * error;
                }
            }
        }

        // Step 4: Update theta using the gradient
        for (int p = 0; p < n; ++p) {
            for (int j = 0; j < k; ++j) {
                theta[p * k + j] -= lr * grad_theta[p * k + j] / current_batch_size;
            }
        }
    }

    // Free allocated memory
    delete[] logits;
    delete[] probs;
    delete[] grad_theta;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
