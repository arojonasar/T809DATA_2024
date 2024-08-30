# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import matplotlib.pyplot as plt
import numpy as np

from tools import scatter_2d_data, bar_per_axis


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    std: np.float64
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    # Identity matrix is done with np.eye
    idMatrix = np.eye(k)
    # Variance is standard deviation squared
    variance = np.square(std)
    cov_matrix = variance * idMatrix

    return np.random.multivariate_normal(mean, cov_matrix, n)


def visualize_data():
    data = gen_data(300, 2, [-1,2], np.sqrt(4))
    scatter_2d_data(data)
    bar_per_axis(data)


def update_sequence_mean(
    mu: np.ndarray, # the mean vector (old_mean)
    x: np.ndarray, # new data point
    n: int # current step
) -> np.ndarray: # returns the updated mean vector
    '''Performs the mean sequence estimation update
    '''
    # The formula is:
    # new_mean = old_mean + 1/n (x-old_mean)
    return mu + (1/n)*(x-mu)


def _plot_sequence_estimate():
    data = gen_data(100, 2, [-1, 2], 3)
    estimates = [np.array([0, 0])]

    for i in range(data.shape[0]):
        x = data[i]
        new_mean = update_sequence_mean(estimates[0], x, i+1)
        estimates.append(new_mean)

    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.legend(loc='upper center')
    plt.savefig('4_1.png')
    plt.show()


def _square_error(y, y_hat):
    return np.square(y - y_hat)


def _plot_mean_square_error():
    true_mean = np.array([-1, 2]) # Let the true mean be (-1, 2)
    mean_prediction = [0,0] # Start with zero as a prediction, update this value as we go
    data = gen_data(100, 2, true_mean, 3)
    mseArr = [] # Mean square error array

    for i in range(data.shape[0]):
        x = data[i]
        mean_prediction = update_sequence_mean(mean_prediction, x, i+1)
        mean_square_error = np.mean( _square_error(true_mean, mean_prediction))
        mseArr.append(mean_square_error)

    plt.plot(mseArr)
    plt.savefig('5_1.png')
    plt.show()


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """   
    # Section 1 
    print(gen_data(5, 1, np.array([0.5]), 0.5))
    
    # Section 2
    visualize_data()

    # Section 3
    X = gen_data(1, 2, np.array([1, 1]), 0.5)
    mean = np.mean(X, 0)
    new_x = gen_data(1, 2, np.array([0, 0]), 1)
    return3 = update_sequence_mean(mean, new_x, X.shape[0]+1)
    print(return3)

    # Section 4
    _plot_sequence_estimate()

    # Section 5
    _plot_mean_square_error()
