import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
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
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    num_train,dim = X.shape #N,D
    num_classes = W.shape[1] #C

    for i in range(num_train):
        f_x = X[i].dot(W)
        f_x -= np.max(f_x)
        prob = np.exp(f_x) / np.sum(np.exp(f_x))
        loss += -np.log(prob[y[i]])

        prob[y[i]] -= 1 
        for j in range(num_classes):
            dW[:,j] += X[i,:] * prob[j]

    loss /= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
  
    #############################################################################
    #                     END OF YOUR CODE                                      #
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    num_train,dim = X.shape #N,D
    num_classes = W.shape[1] #C
    f_x = X.dot(W) 
    f_x -= np.max(f_x,axis=1, keepdims=True)
    prob = np.exp(f_x) / np.sum(np.exp(f_x),axis=1,keepdims=True)
    correct_prob = prob[range(num_train),y]

    loss = np.sum(-np.log(correct_prob)) / num_train
    loss += 0.5 * reg * np.sum(W*W) 

    prob[range(num_train),y] -= 1
    dW = X.T.dot(prob) / num_train

    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW
