from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        std = weight_scale
        self.params['W1'] = std * np.random.randn(input_dim, hidden_dim) # return a matrix with size (input_size, hidden_size), these initial values are drawn from a standard normal distribution                                                                      
        self.params['b1'] = np.zeros(hidden_dim) 
        self.params['W2'] = std * np.random.randn(hidden_dim, num_classes) 
        self.params['b2'] = np.zeros(num_classes)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****     
        
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
     
        out1, cache1          = affine_forward(np.asarray(X), W1, b1)           # affine
        out1_relu, cache_relu = relu_forward(out1)                              # Relu
        scores, cache2        = affine_forward(np.asarray(out1_relu), W2, b2)   # affine
        
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)  # calculate softmax loss, data loss
         
        dout1_relu, dW2, db2 =  affine_backward(dscores, cache2) #(out1_relu, W2, b2))  # back prop for the second affine
        dout1                =  relu_backward(dout1_relu, cache_relu) #out1_relu)  # back pro for relu
        dx, dW1, db1         =  affine_backward(dout1, cache1) # (np.asarray(X), W1, b1))  # back prop for the first affine

        reg = self.reg
        # add regularization term, weight loss
        loss += 0.5*(reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2) )
        
        
        dW1   += 0.5*reg*2*W1 
        dW2   += 0.5*reg*2*W2 
        #db1   += 0.5*reg*2*b1 
        #db2   += 0.5*reg*2*b2 
        
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes
        : An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1    # if the input dropout is not 1, then use droput
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        std = weight_scale
        
        # INITIALIZE THE PARAMETERS OF W AND b FOR EACH LAYER
        for n in range(self.num_layers):            
            # initialize the weight parameter, W1, W2, W3 ...           
            # for the first layer
            if n == 0:  # the first input, the input size is in 'input_dim'
                self.params['W1'] = std * np.random.randn(input_dim, hidden_dims[n]) # return a matrix with size (input_size, hidden_size), these initial values are drawn from a standard normal distribution                                                                                         
                self.params['b1'] = np.zeros(hidden_dims[n]) 

            # for the last layer
            elif n == self.num_layers-1:  # the last layer there is no hidden dimension, but the output size is the number of classes                
                self.params['W'+str(n+1)] = std * np.random.randn(hidden_dims[n-1], num_classes)   # number of layers is one more than number of hidden layers
                self.params['b'+str(n+1)] = np.zeros(num_classes) 

            # For non-first and non-last layers
            else: # the second layer is using the first hidden_dims as input size, the third layer is using the second hidden_dims ...
                self.params['W'+str(n+1)] = std * np.random.randn(hidden_dims[n-1], hidden_dims[n]) # return a matrix with size (input_size, hidden_size), these initial values are drawn from a standard normal distribution                                                                                         
                self.params['b'+str(n+1)] = np.zeros(hidden_dims[n]) 

   
        # INITIALIZE THE SCALING AND SHIFTING PARAMETERS FOR BATCH NORMALIZATION 
        # if batchnormalization is called, then initialize the batchnormalization parameters
        # these parameters are only used during batchnorm, so the iteration of layers is just 
        #   for number of hidden layers, not for all layers. 
        
        if self.normalization == 'batchnorm':            
           num_hidden = len(hidden_dims)  # number of hidden layers            
           for n in range(num_hidden):   
               
               '''  
               if n == num_hidden-1:  # last layer
                   self.params['gamma'+str(n+1)] = np.ones(num_classes)
                   self.params['beta'+str(n+1)]  = np.zeros(num_classes)
               
                else:  # non-last layer
                '''
               self.params['gamma'+str(n+1)] = np.ones(hidden_dims[n])
               self.params['beta'+str(n+1)]  = np.zeros(hidden_dims[n])

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]
            

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # iterate through all the layers that will need to apply BatchNorm
               
        cache_list = []  # create a cache list for later use
        dropout_cache_list = []
        for i in range(self.num_layers):    
            W = self.params['W'+str(i+1)]
            b = self.params['b'+str(i+1)]               
            
            # the very first layer  
            if i == 0:                           
                if self.normalization=='batchnorm':       # with batchnorm        
                    gamma = self.params['gamma'+str(i+1)]
                    beta  = self.params['beta'+str(i+1)]
                    out, cache = affine_batchnorm_relu_forward(X, W, b, gamma, beta, self.bn_params[i])                
                else: # no normalization
                    out, cache = affine_relu_forward(X, W, b)      
                    
                if self.use_dropout:  # if decide to use dropout layer
                   out, dropout_cache = dropout_forward(out, dropout_param)
                   dropout_cache_list.append(dropout_cache) 
                   
            # the last layer
            elif i == self.num_layers-1 :  
                # the last layer is simply a fully connected layer, no BatchNorm
                scores, cache = affine_forward(out, W, b)
            
            
            # non-first and non-last layer 
            else:                 
                if self.normalization=='batchnorm':       # with batchnorm  
                    gamma = self.params['gamma'+str(i+1)]
                    beta  = self.params['beta'+str(i+1)]
                    out, cache = affine_batchnorm_relu_forward(out, W, b, gamma, beta, self.bn_params[i])
                else: # no normalization
                    out, cache = affine_relu_forward(out, W, b)               
            
                if self.use_dropout:  # if decide to use dropout layer
                    out, dropout_cache = dropout_forward(out, dropout_param)
                    dropout_cache_list.append(dropout_cache) 
            
            cache_list.append(cache)  
                      
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)  # calculate softmax loss, data loss
         
        # backpro through each layer to get the derivative. 
        
        all_weight_loss = 0   # initialize a parameter to record all the loss from weight
        reg = self.reg
        
        for i in range(self.num_layers):          
            layerID = self.num_layers - i # the ID to indicate which layer this is, ID is from 1 to num_layers            
            W = self.params['W'+str(layerID)]
            #b = self.params['b'+str(layerID)]
            
            # the first time backpro the last layer, which is a affine layer  
            if layerID == self.num_layers:                
                dout, dW, db =  affine_backward(dscores, cache_list[layerID-1])  # back prop for the second affine

            # backpro the rest of the layers, which is affine-batchnorm-relu convenience layer
            else: 
                if self.use_dropout:  # if decide to use dropout layer
                   dout = dropout_backward(dout, dropout_cache_list[layerID-1])
                
                if self.normalization=='batchnorm':       # with batchnorm              
                    dout, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, cache_list[layerID-1]) # layerID is one number bigger than the index
                    #dout, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(d_dropout, cache_list[layerID-1]) # layerID is one number bigger than the index
                    grads['gamma'+str(layerID)] = dgamma  # store the gradient for bias b for a particular layer
                    grads['beta'+str(layerID)]  = dbeta  # store the gradient for bias b for a particular layer
                else:  
                                        #dx, dW, db =  affine_relu_backward(d_dropout, cache_list[layerID-1])
                    dout, dW, db =  affine_relu_backward(dout, cache_list[layerID-1])
                    
       
            
            all_weight_loss +=np.sum(W*W)   # collect the loss caused by the weight parameter
            dW  += 0.5*reg*2*W   # for a W in a layer, update the L2 regularization       
            grads['W'+str(layerID)]     = dW  # store the gradient for weights W for a particular layer
            grads['b'+str(layerID)]     = db  # store the gradient for bias b for a particular layer

   
        # add regularization term for loss
        loss += 0.5*(reg * all_weight_loss)
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
