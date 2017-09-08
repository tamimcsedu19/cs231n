import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:,j] += (X[i]/num_train)
        dW[:,y[i]] -= (X[i]/num_train)
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_score_vector = scores[np.arange(num_train),y].reshape(num_train,1)
  margins = scores - correct_class_score_vector
  margins += 1
  margins[np.arange(X.shape[0]), y] = 0
  ind_loss = np.maximum(0,margins)

  loss = np.sum(ind_loss) / num_train + reg * np.sum(W * W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  dW += (X.T.dot(ind_loss > 0) / num_train)

  zeros = np.zeros((X.shape[0],W.shape[1]))

  zeros[np.arange(X.shape[0]), y] = (ind_loss > 0).sum(axis = 1)

  dW -= (X.T.dot(zeros) / num_train)
  dW += 2 * reg * W
  # dind_loss = np.ones(ind_loss.shape) / num_train
  # dmargin = (ind_loss > 0)* dind_loss
  # ones = np.ones(margins.shape)
  # ones[np.arange(X.shape[0]), y] = 0
  # dmargin = ones * dmargin
  #
  # dcorrect_class_score_vector = -1 * np.ones(dmargin.shape) * dmargin
  #
  # dscores = dmargin
  # zeros = np.zeros(scores.shape)
  # zeros[np.arange(num_train),y] = 1
  # dscores += (zeros * dcorrect_class_score_vector)
  #
  # dW = X.T.dot(dscores) + reg * np.sum(W * W)




  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
