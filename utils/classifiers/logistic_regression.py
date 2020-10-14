import numpy as np
from random import shuffle

def sigmoid(x):
    #h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return 1.0 / (1 + np.exp(-x))

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)
      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.
      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength
      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    
    y = get_one_hot(y,2)
    num_train,dim = X.shape #N,D
    num_classes = W.shape[1] #C

    for i in range(num_train):
        f_x = X[i,:].dot(W)
        h_x = sigmoid(f_x)
        loss += y[i].dot(np.log(h_x)) + (1-y[i]).dot(np.log(1-h_x))
        dW += X[i,:].reshape(dim,1).dot((h_x-y[i]).reshape(1,2))
    loss = -loss
    loss /= num_train
    loss += reg/(2 * num_train) * np.sum(W * W)
    dW  /= num_train
    dW  += reg * W 
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    
    num_train,dim = X.shape #N,D
    y = get_one_hot(y,2)
    f_x_mat = X.dot(W) 
    h_x_mat = sigmoid(f_x_mat)
    loss = -np.sum(y * np.log(h_x_mat) + (1 - y) * np.log(1 - h_x_mat))/num_train
    loss += reg/(2 * num_train) * np.sum(W * W)
    dW = X.T.dot(h_x_mat - y)/num_train
    dW += reg/num_train * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    

    return loss, dW