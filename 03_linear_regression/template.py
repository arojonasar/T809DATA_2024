# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import torch
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float
) -> torch.Tensor:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    
    N, D = features.shape
    M = mu.shape[0]

    # We force all covariance matrices to be identical and diagonal
    cov = var * torch.eye(D)

    # Initialize the output
    fi = torch.zeros((N, M))

    # Convert the parameters to numpy to use with scipy multivariate_normal function
    np_features = features.numpy()
    np_cov = cov.numpy()

    for m in range(M):
        np_mu = mu[m].numpy()
        # Create multivariate normal distributions from the m-th mu and covariance matrix
        mvn = multivariate_normal(mean=np_mu, cov=np_cov)

        # Calculate the PDF for each feature, convert it back to pytorch tensor
        # and store it in m-th column of the output tensor fi
        fi[:, m] = torch.tensor(mvn.pdf(np_features))

    return fi


def _plot_mvn():
    '''
    Plot the output of each basis function, using the same 
    parameters as above, as a function of the features. 
    You should plot all the outputs onto the same plot. 
    '''
    X, t, mu, var = _generate_mvn_params()
    fi = mvn_basis(X, mu, var)
    M = mu.shape[0]

    for m in range(M):
        plt.plot(fi[:, m])

    plt.savefig('2_1.png')
    plt.show()


def max_likelihood_linreg(
    fi: torch.Tensor,
    targets: torch.Tensor,
    lamda: float
) -> torch.Tensor:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''

    M = fi.shape[1]
    
    # The formula: wml = (lamda*I + fi^T * fi)^-1 * fi^T *t

    # lamda*I
    lamda_identity = lamda * torch.eye(M) 
    
    # fi.T is the transpose of fi
    # @ does a matrix multiplication
    fi_T_fi = fi.T @ fi # fi^T * fi
    fi_T_targets = fi.T @ targets # fi^T * t

    # (lamda*I + fi^T * fi)^-1
    inverse = torch.inverse(lamda_identity + fi_T_fi)

    return inverse @ fi_T_targets


def linear_model(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    # Get basis functions for the features
    fi = mvn_basis(features,mu,var)

    # Return the predictions which are computed by multiplying the basis functions with the weigths
    return fi @ w

def _generate_mvn_params():
    # X are features, t are targets
    X, t = load_regression_iris()
    N, D = X.shape

    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    return X, t, mu, var


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    X, t, mu, var = _generate_mvn_params()
    print('t: ', t)

    fi = mvn_basis(X, mu, var)
    # _plot_mvn()

    fi = mvn_basis(X, mu, var)
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    print('wml: ', wml)

    prediction = linear_model(X, mu, var, wml)
    print('predictions: ' ,prediction)

    # Calculate the Mean Squared Error
    mse = torch.mean((prediction - t) ** 2)
    print('mse: ', mse)

    plt.scatter(t, prediction, alpha=0.5, label='Predictions')
    plt.plot([t.min(), t.max()], [t.min(), t.max()], 'r--', label='Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.savefig('5_a.png')
    # plt.show()

    residuals = prediction - t
    plt.scatter(t, residuals, alpha=0.5, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.legend()
    plt.savefig('5_b.png')
    # plt.show()

