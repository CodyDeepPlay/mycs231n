from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange
import math

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
    dW = np.zeros_like(W)  # size(3073, 10)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train   = X.shape[0]

    for i in range(num_train):        
        # For a given image, calculate the loss between all other classes and its correct class
        dW_one_traing = np.zeros_like(W)  # size(3073, 10), just for one training example

        scores     = X[i].dot(W)      # an individual image dot product the weight matrix, giving the scores
        scores    -= np.max(scores)   # normalize the data by minusing the maximum number, for calculation stability later
        exp_scores =  np.exp(scores)  
        
        sum_score = np.sum(exp_scores)  # sum of exp_score for each training example
        
        for j in range(num_classes):            
            dW_one_traing[:,j] += exp_scores[j]*X[i] # part of the gradient caused by incorrect classes

            # for correct class
            if j == y[i]:                             
               correct_class_loc = j # record which class is the correct class for a particular image
                
        loss += -np.log(exp_scores[correct_class_loc]/sum_score)
        dW_one_traing /= sum_score  # for a particular training example, divide it with the exp score of all classes
        dW_one_traing[:,correct_class_loc] -= X[i] # for correct class, there is an extra term
        
        dW += dW_one_traing
     
    loss /= num_train            # average over all the training examples
    loss += reg * np.sum(W * W)  # don't forget the regularization term
    dW   /= num_train            # Add regularization to the parameter gradient. 
    dW   += reg*2*W              # update the gradient for regularization term, check the definition for loss function, 


    
    pass

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
    num_train = X.shape[0]
    length = X.shape[1]  # feature length of each training example
    num_class = dW.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores_matx  = X.dot(W)                    # size of(500 ,10), where X(500, 3073), W(3073, 10)
    max_each_ob  = np.max(scores_matx, 1)      # for each observation, what is the maximum score
    scores_matx -= max_each_ob[:,np.newaxis]   # in each score, minus the maximum, for later calculation stability
    exp_scores_matrx   = np.exp(scores_matx)   # size of (500, 10)

    across_all_examples  = np.arange(num_train)  # generate an index vector for acrossing all the training examples.
    correct_class_scores = exp_scores_matrx[across_all_examples, y] # get the score for the correct class in each training example
     
    normalized_prob =-np.log( correct_class_scores/np.sum(exp_scores_matrx, 1) )
    loss = np.sum(normalized_prob) / num_train
    loss += reg * np.sum(W * W)  # don't forget the regularization term

    
    # Gradient term caused by the correct class only in each training example
    dW_3D = np.zeros((num_train,length,num_class)) # size (500, 3073, 10)
    dW_3D[across_all_examples,:,y] -= X[across_all_examples,:]  # assign the extra term in the gradient for correct class for all the training example
    dW += np.sum(dW_3D,0)  # update the gradient caused by the correct classes across all the training examples

    # Gradient term caused by the all class (correct and incorrect classes) in each training example
    #dW_exp_X = exp_scores[:,:,np.newaxis]*(X[:,np.newaxis,:])
    dW_exp_3D = exp_scores_matrx[:,np.newaxis,:]*(X[:,:,np.newaxis])
    
    sum_scores = np.sum(exp_scores_matrx,1) # one sum scoreacross all classes for each training example
    dW_exp_3D /= sum_scores[:,np.newaxis,np.newaxis] # divide the summation scores across all the training examples
    
    # update the gradient caused by all classes across all training examples, 
    # NOTE: without the extra terms from correct class, which has already updated previously
    dW +=  np.sum(dW_exp_3D,0)   
    
    dW /= num_train  # Add regularization to the parameter gradient. 
    dW += reg*2*W    # update the gradient for regularization term, check the definition for loss function, 
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW




















