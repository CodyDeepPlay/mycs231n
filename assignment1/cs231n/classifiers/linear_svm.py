from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange

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
   # dW = np.zeros(W.shape) # initialize the gradient as zero
    dW_new = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train   = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        #single_loss = 0.0       # initiate the loss for a single image
        num_margins_bigger_0 = 0  # number of margins that is bigger than 0
        # For a given image, calculate the loss between all other classes and its correct class
        scores = X[i].dot(W)                # an individual image dot product the weight matrix, giving the scores
        correct_class_score = scores[y[i]]  # among the scores we got, which one is the score for the correct class
        for j in range(num_classes):

            # for correct class
            if j == y[i]:                # the correct class
                correct_class_loc = j    # for this given image, update the correct class location in the scores vector  
                continue                 # return to the beginning of the for loop
            
            margin = scores[j] - correct_class_score + 1 # note delta = 1

            # for incorrect class
            #margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0: 
                loss += margin  # update the cumulated loss for all images so far, if margin is bigger than 0   
                #single_loss += margin    # update the loss for a single image

                # update the gradient of parameter matrix for non-correct class
                #dW[:,j] += margin*X[i] # when margin>0, what is the gradient
                dW_new[:,j] +=X[i]
                
                num_margins_bigger_0 +=1
            #else:  dW[:,j] += 0*X[i]   # when margin<0, what is the gradient
                        
       # dW[:,correct_class_loc] += -single_loss*X[i] # update the gradient of parameter matrix for the correct class  
        dW_new[:,correct_class_loc] -= num_margins_bigger_0*X[i]       
        
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # the same operation to parameter gradient
    #dW   /= num_train  # Add regularization to the parameter gradient. 
   # dW += reg*2*W      # update the gradient for regularization term, check the definition for loss function, 
    
    dW_new /= num_train
    dW_new += reg*2*W 
    
    dW = dW_new
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    all_scores   = X.dot(W)    # all the scores
    num_train    = X.shape[0]  # number of training examples
    all_training_example = np.arange(num_train)
     
    correct_class_scores = all_scores[all_training_example, y]         # all the scores for the correct class    
   
    all_margins = all_scores - correct_class_scores[:,np.newaxis] + 1  # [500, 10], 500 training examples, and 10 classes each    
    all_margins[all_training_example, y] = 0    # the correct class has no margin, set them to 0

    # only consider margins that is bigger than 0
    all_margins [np.where(all_margins<=0)]= 0  # only count where the margins bigger than 0 
    
    
    
    # CALCULATE THE LOSS
    # accumulate all the margin>0, and also remove the fake margins caused by correct class itself
    loss  = np.sum( all_margins ) /num_train  # based on the definition, loss should be the average
    loss += reg * np.sum(W * W)  # udpate the regularization term 
    
    
    # only consider when margins are bigger than 0
    all_margin_loc_matrx = np.zeros_like(all_margins)
    all_margin_loc_matrx[np.where(all_margins>0)] = 1  # indicate where margins bigger than 0, size (500, 10)
      
    # assign all the gradient caused by incorrect classes    
    dW_3D = all_margin_loc_matrx[:,np.newaxis,:] * X[:,:,np.newaxis] # size(500, 3073, 10)
    
    
    # calculate the gradient caused by correct class  # size(500, 3073)
    
    dW_3D[all_training_example, :, y] -= np.sum(all_margin_loc_matrx,1)[:,np.newaxis] * X
    
    dW = np.sum(dW_3D,0)
    
    '''
    # CALCULATE THE GRADIENT
    # the weight gradient caused by non-correct classes
    all_weight_grad_0  = X[:,:,np.newaxis]*all_margins[:,np.newaxis,:]  # [500, 3073, 10] # weights gradient by all the training classes
    
    ### MINGMING CONTINUE WORKING HERE
    
    # the weight gradient caused by correct classes
    all_single_loss   = np.sum(all_margins,1)     # all the single loss for each image
    all_weight_grad_1 = -X * all_single_loss[:, np.newaxis] # [500, 3073], these are the gradient only for the correct classes
    # add the correct class and non-ccorrect class gradient together,
    # NOTE: only add to the location of the correct classes in the 'non-correct class gradient'
    all_weight_grad_0[all_training_example,:,y[all_training_example]] = all_weight_grad_1  
    dW = np.sum(all_weight_grad_0, axis=0)/num_train  # add the gradient from each training example together, and make it average
    '''
    dW /= num_train
    dW += reg*2*W      # update the gradient for regularization term, check the definition for loss function, 


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
