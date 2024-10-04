from typing import Union
import torch
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    # The formula is 1 / (1 + e^-x)
    exp = torch.exp(-x)
    return 1 / (1 + exp)


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    # The formula is σ(a)(1−σ(a))
    s = sigmoid(x)
    return s * (1-s)


def perceptron(
    x: torch.Tensor,
    w: torch.Tensor
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    multi = x @ w
    return multi, sigmoid(multi)


def ffnn(
    x: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    
    Parameters:
        x  : input pattern (1 × D), a row vector.
        M  : number of hidden neurons.
        K  : number of output neurons.
        W1 : weight matrix for input to hidden layer, size (D + 1) × M.
        W2 : weight matrix for hidden to output layer, size (M + 1) × K.
        
    Returns:
        y  : output of the network, size (1 × K).
        z0 : input vector (1 × (D + 1)), input with bias term.
        z1 : output of hidden layer (1 × (M + 1)), with bias term.
        a1 : input vector to the hidden layer (1 × M), before activation.
        a2 : input vector to the output layer (1 × K), before activation.
    '''

    # Ensure that the input vector is of the right size by reshaping it to (1 x D) 
    if x.dim() == 1:
        # The -1 here makes PyTorch figure out the dimensions based on the number of elements in x
        x = x.view(1, -1) 

    # Bias term
    b = torch.ones((x.shape[0], 1))
    
    # z0: adding the bias term in front of the input (N x (D + 1))
    input_biased = torch.cat((b, x), dim=1)  

    # a1: the input to the hidden layer (N x M)
    hidden_input = input_biased @ W1
   
    # apply activation function to get hidden layer output
    hidden_output = sigmoid(hidden_input)

    # z1: add bias term to hidden output (N x (M + 1))
    hidden_biased = torch.cat((b, hidden_output), dim=1)

    # a2: the input to the output layer (N x K)
    output_input = hidden_biased @ W2

    # y: final output (N x K)
    output = sigmoid(output_input)

    return output, input_biased, hidden_biased, hidden_input, output_input


def backprop(
    x: torch.Tensor,
    target_y: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    # Run ffnn on the input
    output, input_biased, hidden_biased, hidden_input, _ = ffnn(x, M, K, W1, W2)

    # Calculate the error for the output layer
    delta_k = output - target_y
    
    # Calculate the delta_j for the hidden layer
    # First remove the bias (first row) from W2
    W2_no_bias = W2[1:, :]  # (M x K) without the bias
    delta_j = d_sigmoid(hidden_input) * (delta_k @ W2_no_bias.T)

    # Initialize dE1 and dE2 as zero-matrices with the same shape as the weights
    dE1 = torch.zeros(W1.size())
    dE2 = torch.zeros(W2.size())

    dE1 = input_biased.T @ delta_j
    dE2 = hidden_biased.T @ delta_k

    return output, dE1, dE2


def train_nn(
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
    iterations: int,
    eta: float
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    # Initialize necessary variables
    N = X_train.size(0)
    Etotal = torch.zeros(iterations)
    misclassification_rate = torch.zeros(iterations)

    # One-hot encode the targets so it matches the output of backprop
    t_train = one_hot_encode(t_train.long(), K)

    # Run a loop for iterations iterations.
    for i in range(iterations):
        # In each iteration we will collect the gradient error matrices for each data point. 
        # Start by initializing dE1_total and dE2_total as zero matrices with the same shape as W1 and W2 respectively.
        dE1_total = torch.zeros_like(W1)
        dE2_total = torch.zeros_like(W2)
        last_guesses = torch.empty(N, K)
        total_error = 0.0

        # Run a loop over all the data points in X_train. 
        # In each iteration we call backprop to get the gradient error matrices and the output values.
        for j in range(N):
            x = X_train[j]
            target_y = t_train[j]

            output, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2

            # Store the probabilities in last_guesses
            last_guesses[j] = output.squeeze() # Squeeze to ensure it's (K,) rather than (1, K)

            # Calculate cross-entropy error
            cross_entropy = -torch.sum(target_y * torch.log(output) + (1 - target_y) * torch.log(1 - output))
            total_error += cross_entropy.item()

        # Once we have collected the error gradient matrices for all the data points, we adjust the weights in W1 and W2
        # using W1 = W1 - eta * dE1_total / N where N is the number of data points in X_train (and similarly for W2)
        W1 = W1 - eta * dE1_total / N
        W2 = W2 - eta * dE2_total / N

        Etotal[i] = total_error / N

        # Calculate misclassification rate
        predicted_classes = torch.argmax(last_guesses, dim=1)
        true_classes = torch.argmax(t_train, dim=1)
        misclassification_rate[i] = (predicted_classes != true_classes).float().mean()

    # When the outer loop finishes, we return from the function
    return W1, W2, Etotal, misclassification_rate, predicted_classes

def one_hot_encode(
    targets: torch.Tensor, 
    num_classes: int
) -> torch.Tensor:
    # Create a zero matrix of shape (N, K)
    one_hot = torch.zeros(targets.size(0), num_classes)
    
    # Use scatter_ to fill in the one-hot encoding (1 at the index of the target class)
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    
    return one_hot

def test_nn(
    X: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> torch.Tensor:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    # Number of test samples
    N = X.shape[0]
    predictions = torch.zeros(N)

    # Forward propagate each sample through the network
    for i in range(N):
        x = X[i, :]
        output, _, _, _, _ = ffnn(x, M, K, W1, W2)

        # Get the index (class) with the maximum output probability
        predicted_class = torch.argmax(output, dim=1)

        # Store the predicted class
        predictions[i] = predicted_class.item()

    return predictions


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """

    """Section 1.1"""
    # print(sigmoid(torch.Tensor([0.5])))
    # print(d_sigmoid(torch.Tensor([0.2])))

    """Section 1.2"""
    # print(perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1])))
    # print(perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4])))

    """Section 1.3"""
    # initialize the random generator to get repeatable results
    torch.manual_seed(4321)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)

    # initialize the random generator to get repeatable results
    torch.manual_seed(1234)

    # Take one point:
    x = train_features[0, :]
    K = 3  # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    # print('y:', y)
    # print('z0:', z0)
    # print('z1:', z1)
    # print('a1:', a1)
    # print('a2:', a2)

    """Section 1.4"""
    # initialize random generator to get predictable results
    torch.manual_seed(42)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    x = features[0, :]

    # create one-hot target for the feature
    target_y = torch.zeros(K)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

    # print('y:', y)
    # print('dE1:', dE1)
    # print('dE2:', dE2)
    
    """Section 2.1"""
    # initialize the random seed to get predictable results
    torch.manual_seed(1234)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    
    # print('W1tr:', W1tr)
    # print('W2tr:', W2tr)
    # print('Etotal:', Etotal)
    # print('misclassification_rate:', misclassification_rate)
    # print('last_guesses:', last_guesses)
    
    """Section 2.2"""
    M = 6 # Size of hidden layer
    K = 4 # Size of output layer
    guesses = test_nn(features, M, K, W1tr, W2tr)

    # print(guesses)
