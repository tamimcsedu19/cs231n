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

    num_train = X.shape[0]
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    loss = 0
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= scores.max()
        loss += -scores[y[i]] + np.log(np.exp(scores).sum())
        dW[:, y[i]] -= (X[i] / num_train)

        tempdW = X[i].reshape(X[i].shape[0],1).dot(np.exp(scores).reshape(1,scores.shape[0]))
        tempdW /= np.exp(scores).sum()
        tempdW /= num_train
        dW += tempdW

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    scores = scores - scores.max(axis = 1).reshape(scores.shape[0],-1) #numeric stability
    exp_scores = np.exp(scores)
    scores_sum = np.sum(exp_scores, axis = 1).reshape(exp_scores.shape[0],-1)
    probabilities = exp_scores / scores_sum
    loss = np.sum(-np.log(probabilities[np.arange(X.shape[0]),y])) / X.shape[0] + reg * np.sum(W * W)

    correctClassMat = np.zeros((num_train,num_class))
    correctClassMat[np.arange(num_train),y] = 1

    dW += (X.T.dot(probabilities)/num_train + 2 * reg * W - X.T.dot(correctClassMat)/num_train)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
