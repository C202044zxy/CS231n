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

    N, D = X.shape 
    C = W.shape[1]
    for i in range(N): # for every test
        z = np.zeros(C)
        for j in range(C):
            z[j] = np.exp(np.dot(X[i], W[:, j])) 
        sum = np.sum(z) 
        z /= sum
        loss -= np.log(z[y[i]]) 
        for j in range(C):
            if j == y[i]: # (z[j]-1)
                dW[:, j] += (z[j] - 1) * X[i]
            else : # (z[j])
                dW[:, j] += z[j] * X[i]
    # dW : D*C ; z : N*C ; X : N*D
    loss /= N
    dW /= N
    loss += reg * np.sum(W**2)
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

    """
    First step: z = X * W, covert z to the probability matrix. 
    Then: compute the loss (loss = \sum -\log z[yi])
    Finally: compute the gradient. 
    """
    
    z = np.exp(np.matmul(X, W))
    sum = np.sum(z, axis = 1)
    z = z / sum.reshape(-1, 1)

    N = X.shape[0]
    loss = - np.log(np.sum(z[np.arange(N), y]))
    dW = np.matmul(X.T, z)
    for i in range(N):
        dW[:, y[i]] -= X[i]
    dW /= N
    loss /= N
    
    loss += reg * np.sum(W**2) 
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
