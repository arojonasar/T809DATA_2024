# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def gen_data(
    n: int, # Number of data points
    locs: np.ndarray, # Means
    scales: np.ndarray # Standard deviations
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    '''
    features = [] # Data points
    targets = [] # Corresponding classes
    classes = [i for i in range(len(locs))]

    for i in range(len(locs)):
        data = norm.rvs(loc=locs[i], scale=scales[i], size=n)
        features.extend(data)
        # Since the mean (locs[i]) is the same, we're working with the same distribution every iteration
        # That's why we can just add "i" n-times to the list of targets
        targets.extend([i] * n)

    # Later functions expects features and targets to be np arrays
    features = np.array(features)
    targets = np.array(targets)
    return features, targets, classes


# Helper function that returns only features of selected class
def _find_relevant_features(
        features: np.ndarray,
        targets: np.ndarray,
        selected_class: int
) -> list:
    relevant_features = []
    for i in range(len(targets)):
        if targets[i] == selected_class:
            relevant_features.append(features[i])
    return relevant_features


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    relevant_features = _find_relevant_features(features, targets, selected_class)
    return np.mean(relevant_features)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    relevant_features = _find_relevant_features(features, targets, selected_class)
    return np.cov(relevant_features, rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return norm.pdf(feature, loc=class_mean, scale=class_covar)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        # Find mean and covariance of all classes
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))

    likelihoods = []
    for i in range(test_features.shape[0]):
        # Find the likelihood for each class
        class_likelihood = []
        for j in range(len(classes)):
            class_likelihood.append(likelihood_of_class(test_features[i], means[j], covs[j]))

        likelihoods.append(class_likelihood)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, 1)


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    print('----- Section 1 -----')
    # Generate 50 samples from two normal distributions: N(-1, sqrt(5)) and N(1, sqrt(5))
    features, targets, classes = gen_data(50, [-1, 1], [np.sqrt(5), np.sqrt(5)])

    # features, targets, classes = gen_data(10, [-4, 4], [np.sqrt(2), np.sqrt(2)])

    # Create a train and test set using split_train_set from tools with 80% train and 20% test split
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, 0.8)

    print('----- Section 2 -----')
    marker_styles = ['o', 'x', '_', '^', '+']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(features)):
        plt.scatter(features[i], 0, c=colors[targets[i]], marker=marker_styles[targets[i]])
    plt.savefig('2_1.png')
    plt.show()

    print('----- Section 3 -----')
    mean = mean_of_class(train_features, train_targets, 0)
    print(mean)

    print('----- Section 4 -----')
    covar = covar_of_class(train_features, train_targets, 0)
    print(covar)

    print('----- Section 5 -----')
    likely = likelihood_of_class(test_features[0:3], mean, covar)
    print(likely)

    print('----- Section 6 -----')
    maximum = maximum_likelihood(train_features, train_targets, test_features, classes)
    print(maximum)

    print('----- Section 7 -----')
    prediction = predict(maximum)
    print(prediction)
    