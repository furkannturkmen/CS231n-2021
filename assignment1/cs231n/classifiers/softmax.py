from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    m = X.shape[0]
    dim = X.shape[1]
    num_class = W.shape[1]
    for i in range(m):
      score = np.dot(X[i].reshape((dim, 1)).T, W)
      e_score = np.exp(score)
      prob = e_score / np.sum(e_score)
      loss += - np.log(prob[:,y[i]])
      for c in range(num_class):
        if c == y[i]:
          dW[:,c] += X[i] * (prob[:,c] - 1)
          
        else:
          dW[:,c] += X[i] * prob[:,c]

    loss /= m
    loss += reg * np.sum(W * W)


    dW /= m
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    m = len(y)
    num_class = W.shape[1]

    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    prob = exp_scores / np.sum(exp_scores, axis = 1, keepdims=True)
    loss = np.sum(- np.log(prob[np.arange(len(y)),y])) / m
    loss += reg * np.sum(W * W)
    y_one_hot = np.zeros((m, num_class))
    y_one_hot[np.arange(len(y)),y] = 1
    result = prob - y_one_hot
    dW = np.dot(X.T, result)
    dW /= m
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
