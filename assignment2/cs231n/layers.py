#from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dimen = x.shape  # the original dimension of the input data
    X = np.reshape(x, (dimen[0], np.prod(dimen[1:])))   # reshape x into shape [d_1, D]
    
    out = X.dot(w) + b
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    db = np.sum(dout, axis=0)
    
    
    dimen = x.shape  # the original dimension of the input data
    X  = np.reshape(x, (dimen[0], np.prod(dimen[1:])) )   # reshape x into shape (N, D)
    dw = np.dot(X.T, dout)                   # (D, M)
    dx = np.reshape( dout.dot(w.T), dimen )

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x
    out[out<=0]=0
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #d_ReLu = np.zeros_like(x)  # derivative of ReLu function
    
    '''
    https://danieltakeshi.github.io/2017/01/21/understanding-higher-order-local-gradient-computation-for-backpropagation-in-deep-neural-networks/
    refer here for the derivatives of the ReLu function(matrix)
    '''
    '''
    dimen = len(x)
    d_ReLu = np.diag(dimen * [1]) # define a diagonal matrix, based on the size of x
    index = d_ReLu==1  # the original diagonal index where d_ReLu = 1, 
    
    x_pos_index = x[index]>0    # among all the diagonal index, where x>0

    diag = np.zeros(len(x_pos_index))   # recreate the diagnonal vector, 
    diag[x_pos_index] = 1               # set the only where x>0 to be 1, in the diagonal vector
    
    d_ReLu = np.diag(1 *diag) # define a diagonal matrix

    
    #d_ReLu = np.ones_like(x)  # derivative of ReLu function
    #d_ReLu[x<0] = 0   # where x <0, ReLu's derivative is 0
        
    #dx = d_ReLu* dout
    #dx = np.dot(d_ReLu, dout)
    #dout[x<0] = 0
    dx  = d_ReLu*dout   # back propagation
    '''
    dx = dout
    dx[x<=0] = 0
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode     = bn_param['mode']
    eps      = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype)) # if 'running_mean' already exist, return its value
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        sample_mean = np.mean(x, axis=0)    # mean of the current minibatch of data
        sample_var  = np.var(x, axis=0)     # varirance of the current minibatch of data

        x_minus_mean = x - sample_mean  # size of (N,D)
        
        sqrt_var = np.sqrt(sample_var+eps)
        
        div = x_minus_mean/sqrt_var    # normalize the data       
        multi = div * gamma
               
        out = multi + beta  # scale and shift parameter 
        
        # update the running mean and var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var  = momentum * running_var  + (1 - momentum) * sample_var
        

        cache = (sample_mean,  sample_var, 
                 x_minus_mean, sqrt_var, 
                 div, multi, gamma, beta, eps, x)  # will be used during back propagation
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = (x-running_mean)/np.sqrt(running_var+eps)    # normalize the data        
        out = out * gamma + beta                       # scale and shift parameter
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var']  = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N,D) = dout.shape
    ones_matrx = np.ones_like(dout)  # later will use during dirivative calculation 
     
    (sample_mean,  sample_var, 
     x_minus_mean, sqrt_var, 
     div, multi, gamma, beta, eps, x) = cache 
     
    # DRAW THE COMPUTATIONAL GRAPH OF BATCH NORMALIZATION 
    # below are calculation based on the computational graph
     
    # backpro multi + beta = out
    dbeta   = np.sum(dout, axis=0)  # size (D,)      
    dmulti  = 1 *dout               # size (N,D)

    # backpro div*gamma = multi
    ddiv    = gamma*dmulti                 # size(N,D)
    dgamma  = np.sum(div*dmulti , axis=0)  # size(D,)  
    
    # the first branch of x_minus_mean from div
    # backpro x_minus_mean/sqrt_var = div
    dx_minus_mean_div = 1/sqrt_var*ddiv                                          # size(N,D)
    dsqrt_var =np.sum( (x_minus_mean *(-1/(sqrt_var**2)) ) * ddiv,  axis=0 ) # size(D,)
        
    # backpro x-mean = x_minus_mean
    dx    = 1* dx_minus_mean_div                    # size(N,D)
    dmean = np.sum( -dx_minus_mean_div, axis=0)     # size(D,)
    # backpro mean()
    dx += ones_matrx/N *dmean                       # size(N,D)
    
    # continue backpro sqrt(var)
    dvar = 1/(2*np.sqrt(sample_var+eps)) *dsqrt_var  # size(D,)    
    # backpro mean(square)
    dsquare = ones_matrx/N *dvar    
    # backpro square
    dx_minus_mean_sq  = 2*x_minus_mean *dsquare    # size(N,D)
     
    # backpro x-mean = x_minus_mean
    dx  += 1* dx_minus_mean_sq                     # size(N,D)
    dmean_sq = np.sum( -dx_minus_mean_sq, axis=0)  # size(D,)
    # backpro mean()
    dx += ones_matrx/N *dmean_sq  # size(N,D)
    
  
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N,D) = dout.shape
    ones_matrx = np.ones_like(dout)  # later will use during dirivative calculation 
         
    (sample_mean,  sample_var, 
     x_minus_mean, sqrt_var, 
     div, multi, gamma, beta, eps, x) = cache 
    y = div
     
    # backpro y*gamma + beta = out
    dbeta   = np.sum(dout, axis=0)        # size (D,)  
    dgamma  = np.sum(y *dout, axis=0)  # size(D,)    
    
    
    # backpro bacth_norm()
    # from the computational graph with from x through (variance, std_dev, mean) to Y
    # there are three pathways to backpro to input x
    # after hand derive them, it can be write in the following format

       
    '''
    Computational graph in a simpler way
  
    (x)=====================(x-u)=====================(y)
      =                    =    =                   =
        =               =         =              =
          === (u) ====              =(v)====(std)           
    ''' 
    
    '''
    dout_dy = gamma * dout     # size(N, D)
    
    # backpro (y) to (x-u)
    dy_dx_minus_mean = 1/sqrt_var *ones_matrx  * dout_dy    # size (N, D)
    
    # backpro (y) to (std)
    dy_ddelta =   np.sum(-x_minus_mean/(sqrt_var**2) * dout_dy , axis=0) # size(D,)
    # backpro (std) to (v)
    ddelta_dvar =  1/(2*sqrt_var) * dy_ddelta   # size(D,)
    # backrpo (v) to (x-u)
    dvar_dx_minus_mean = 2*x_minus_mean/N*ones_matrx  * ddelta_dvar  # size(N,D)
    
    # notice there are two branches into (x-u), one from (y), one from (v)
    # backpro (x-u) to (x)
    dx_minus_mean_dx  = 1 * (dy_dx_minus_mean + dvar_dx_minus_mean)   # size(N,D)
    
    # backpro (x-u) to (u)
    dx_minus_mean_du  =np.sum( (-1) * (dy_dx_minus_mean + dvar_dx_minus_mean), axis=0)  #size(D,)  
    # backpro (u) to (x)
    du_dx =1/N * ones_matrx* dx_minus_mean_du  # size(N,D)    
   
    dx = (dx_minus_mean_dx + du_dx )  # combine the two branches together
    '''
   
    # hand derive the equation from the computational graph here, 
    # a more detailed implementation is shown above 
    big_part = x_minus_mean/N * np.sum( - x_minus_mean/(sqrt_var**2) * dout, axis=0)    
    double   = gamma/sqrt_var *(dout + big_part)   
    dx       = 1/N*ones_matrx * np.sum(-double, axis=0) + double
   
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (N, D) = x.shape
    sample_mean = np.mean(x, axis=1)    # mean of the current batch of data across all features
    sample_var  = np.var(x, axis=1)     # varirance of the current batch of data across all features

    x_minus_mean = x - sample_mean[:,np.newaxis]  # size of (N,D)
    
    sqrt_var = np.sqrt(sample_var+eps)
    
    y = x_minus_mean/sqrt_var[:,np.newaxis]    # normalize the data       
    multi = y * gamma           
    out   = multi + beta  # scale and shift parameter 


    cache = (sample_mean,  sample_var, 
             x_minus_mean, sqrt_var, 
             y, multi, gamma, beta, eps, x)  # will be used during back propagation
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Refer to backpro for BatchNorm above for more details.
    
    (N,D) = dout.shape
    ones_matrx = np.ones_like(dout)  # later will use during dirivative calculation 
         
    (sample_mean,  sample_var, 
     x_minus_mean, sqrt_var, 
     y, multi, gamma, beta, eps, x) = cache 

    # backpro y*gamma + beta = out
    dbeta   = np.sum(dout, axis=0)        # size (D,)  
    dgamma  = np.sum(y *dout, axis=0)     # size(D,)    
    
    
    # hand derive the equation from the computational graph here, 
    # refere to the notes from batch normalization, which is shown above.

    dout_dy = gamma * dout     # size(N, D)
    
    # backpro (y) to (x-u)
    dout_dx_minus_mean1 = 1/sqrt_var[:,np.newaxis] *ones_matrx  * dout_dy    # size (N, D)
    
    # backpro (y) to (std)
    dout_ddelta =   np.sum(-x_minus_mean/(sqrt_var**2)[:,np.newaxis] * dout_dy , axis=1) # size(N,)
    # backpro (std) to (v)
    dout_dv =  1/(2*sqrt_var) * dout_ddelta   # size(N,)
    # backrpo (v) to (x-u)
    dout_dx_minus_mean2 = 2*x_minus_mean/N*ones_matrx  * dout_dv[:,np.newaxis]  # size(N,D)
    
    # notice there are two branches into (x-u), one from (y), one from (v)
    # backpro (x-u) to (x)
    dout_dx1  = 1 * (dout_dx_minus_mean1 + dout_dx_minus_mean2)   # size(N,D)
    
    # backpro (x-u) to (u)
    dout_du  = np.sum( (-1) * (dout_dx_minus_mean1 + dout_dx_minus_mean2), axis=1)  #size(N,)  
    # backpro (u) to (x)
    dout_dx2 =1/N * ones_matrx* dout_du[:,np.newaxis]  # size(N,D)    
   
    dx = (dout_dx1 + dout_dx2 )  # combine the two branches together

    '''
    double_part1 = (gamma* dout )/sqrt_var[:,np.newaxis]
    double_part2 = (x_minus_mean/(N*sqrt_var[:,np.newaxis]) * 
                    np.sum( -x_minus_mean /( (sqrt_var**2)[:,np.newaxis] )  *  (gamma*dout) ,axis=1)[:, np.newaxis] )
    double =  (double_part1 + double_part2)   
    dx       = 1/N*np.sum(-double, axis=1)[:, np.newaxis] + double
    '''
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta




def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask = np.random.rand(*x.shape) < p   # define the dropout layer mask
        out = x*mask     # apply dropout
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = x
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout*mask
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W) = (num_imgs, img_channels, img_height, img_width)
    - w: Filter weights of shape (F, C, HH, WW)=  (ksize, img_channels, F_height, F_width) 
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    # Iml2col, 
    # turn images into the col vectors for vectorization operations.
    
    (num_imgs, img_channels, img_height, img_width) = x.shape # input data information
    (ksize, img_channels, F_height, F_width)        = w.shape # size information about the filter
    pad    = conv_param['pad']               # the padding parameter
    stride = conv_param['stride']            # the stride parameter
    
    # If following the above pad and stride parameter, what is the output size of the new matrix will be
    # along the height and width dimension.
    newImgHeight =int( ((img_height + 2*pad - F_height) / stride) +1 )
    newImgWidth  =int( ((img_width  + 2*pad - F_width)  / stride) +1 )

    # zero padding the original image
    x_pad = np.zeros((num_imgs, img_channels, img_height+2*pad, img_width+2*pad)) # after zero padding, what is the size should be for the padded images
    x_pad[:,:, pad:img_height+pad, pad:img_width+pad] = x

        
    #------------ THE img2col PROCESS----------------#
    # stretch the image into column matrix
    # when stretch the filter and input data according to the stride and padding information, 
    # the output column matrix should have the size as the following      
    cols = np.zeros((F_height*F_width*img_channels, newImgHeight*newImgWidth*num_imgs))  # output size of the input image after im2col operation    
    
    for num in range(num_imgs): # for each image    
        for heig in range(newImgHeight):  
            # At a given height, iterate through all the widths
            for wid in range(newImgWidth):
                # at a particular location of a given image, what is the part of the image that should be conv with a single filter    
                current_voxel  = x_pad[num, :, 0+heig*stride:F_height+heig*stride, 0+wid*stride:F_width+wid*stride]          
                current_vector = current_voxel.reshape(F_height*F_width*img_channels,)            
                cols[:, heig*newImgWidth + wid + num*newImgWidth*newImgHeight] = current_vector    
    # ------------- End of img2col process --------------# 
    
    conv_param['cols'] = cols   # pass this variable so that we can use it during the back propagation
    
    # Similar operation to the weights
    # Strech the weights into the same way as col2img
    weights = w.reshape(ksize, F_height*F_width*img_channels)    # size of (ksize, img_channels*F_height*F_width)


    # The convolution calculation by using the multiplication of matrix.
    # iterate through all the images to conduct the convolution using maxtrix multiplication
    out_col = np.zeros((ksize, newImgHeight*newImgWidth*num_imgs))  # initialize the space to store the img2col for all the images.
    out_col = weights.dot(cols)    # convolution using matrix multiplication, then add the bias
    
    out_col_b = out_col + b[:,np.newaxis]
    # ----------- The col2img process ----------------#
    # convert the each column back into its 'image' format, following the same order as img2col above
    # iterate though the image width given an image height
    # for heig in range(newImgHeight):        
    #    for wid in range(newImgWidth):

    out_img = np.zeros((num_imgs, ksize, newImgHeight, newImgWidth))   
    for loc in range(newImgHeight*newImgWidth*num_imgs):      
        
        which_img = loc//(newImgHeight*newImgWidth)   # which new image this loc is belong to
        inter_loc = loc%(newImgHeight*newImgWidth)    # which pixel is within the current new image
        
        heig_loc = inter_loc//newImgWidth                            # the height index in the new image
        wid_loc  = inter_loc%newImgWidth                             # the width index in the new image
        out_img[which_img, :, heig_loc, wid_loc] = out_col_b[:, loc]  # with a new image for a certain 'channel' k, 
                                                          # put each individual pixel into its location in the new image.
 
    # ------------- End of col2img process --------------#
    
    '''
    # Anothe way to do it, iterate through each individual image.
    
    # The img2col process
    cols = np.zeros((num_imgs, F_height*F_width*img_channels, newImgHeight*newImgWidth))  # output size of the input image after im2col operation    
    for heig in range(newImgHeight):  
        # At a given height, iterate through all the widths
        for wid in range(newImgWidth):
            # at a particular location, what is the part of the image that should be conv with a single filter    
            current_voxel  = x_pad[:, :, 0+heig*stride:F_height+heig*stride, 0+wid*stride:F_width+wid*stride]          
            current_vector = current_voxel.reshape(num_imgs, F_height*F_width*img_channels)            
            cols[:, :, heig*newImgWidth + wid] = current_vector

    # Similar operation to the weights
    # Strech the weights into the same way as col2img
    weights = w.reshape(ksize, F_height*F_width*img_channels)    # size of (ksize, img_channels*F_height*F_width)

    out_col = np.zeros((num_imgs, ksize, newImgHeight*newImgWidth))  # initialize the space to store the img2col for all the images.
    for num in range(num_imgs):
        out_col[num,:,:] = weights.dot(cols[num,:,:]) + b[:,np.newaxis] 
   
    
    # ----------- The col2img process ----------------#
    
    out_img = np.zeros((num_imgs, ksize, newImgHeight, newImgWidth))    
          
    for loc in range(newImgHeight*newImgWidth):      
        heig_loc = loc//newImgWidth                            # the width index in the new image
        wid_loc  = loc%newImgWidth                             # the width index in the new image
        out_img[:, :, heig_loc, wid_loc] = out_col[:, :, loc]  # with a new image for a certain 'channel' k, 
                                                              # put each individual pixel into its location in the new image.
       
   # out_img = np.reshape(out_col, (num_imgs, ksize, newImgHeight, newImgWidth) )    
   '''
    
    
    out = out_img
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    (num_imgs, img_channels, img_height, img_width) = x.shape # input data information
    (ksize, img_channels, F_height, F_width)        = w.shape # size information about the filter
    cols = conv_param['cols']        # added one of the previous calculated variables, so that no need to calculate it again. 
    pad    = conv_param['pad']       # the padding parameter
    stride = conv_param['stride']    # the stride parameter
    
    (num_imgs, ksize, newImgHeight, newImgWidth) = dout.shape   

    # Reshape dout to get its row major matrix
    dout_col_b = np.zeros((ksize, newImgHeight*newImgWidth*num_imgs))
    for num in range(num_imgs):
                # at a particular location of a given image, what is the part of the image that should be conv with a single filter    
                current_img  = dout[num, :, :, :]          
                current_vector = current_img.reshape(ksize, newImgHeight*newImgWidth)            
                dout_col_b[:, num*newImgHeight*newImgWidth:(num+1)*newImgHeight*newImgWidth] = current_vector
        
    db =  np.sum(dout_col_b, axis=1) #  out_col_b = out_col + b
    dout_col  = dout_col_b
    
    r_weight   = w.reshape(ksize, F_height*F_width*img_channels)   # reshaped weights
    dcols      = (r_weight.T).dot(dout_col)      # derivative of:  out_col  = r_weights * cols
    dr_weights = dout_col.dot(cols.T)            # derivative of:  out_col  = r_weights * cols
     
    # reshape dr_weights back into its original shape
    dw = dr_weights.reshape(ksize, img_channels, F_height, F_width) 


    
    dx_pad = np.zeros((num_imgs, img_channels, img_height+2*pad, img_width+2*pad)) # after zero padding, what is the size should be for the padded images

    for loc in range(dcols.shape[1]): # iterate through each column in the dcols  
        
        which_img = loc//(newImgHeight*newImgWidth)   # which new image this loc is belong to
        inter_loc = loc%(newImgHeight*newImgWidth)    # which pixel is within the current new image
        
        heig_loc = inter_loc//newImgWidth                            # the height index in the new image
        wid_loc  = inter_loc%newImgWidth                             # the width index in the new image
            
        # based on how the original voxel was extracted from x_pad, 
        dx_pad[which_img, :, 0+heig_loc*stride:F_height+heig_loc*stride, 0+wid_loc*stride:F_width+wid_loc*stride] += dcols[:,loc].reshape(img_channels, F_height,F_width)    
    
    dx = dx_pad[:,:,pad:-pad, pad:-pad] 
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N, C, H, W) = x.shape               # the input image dimension 
    pool_w = pool_param['pool_width']    # max pooling parameters
    pool_h = pool_param['pool_height']
    stride = pool_param['stride']

    new_w = int( (W-pool_w)/stride + 1)  # new image width after the max pooling
    new_h = int( (H-pool_h)/stride + 1)  # new image height after the max pooling
    
    out = np.zeros((N, C, new_w, new_h)) # output shape
    out_index = np.zeros((N, C, new_w, new_h)) # track the index of the max number in all images, all channels, but a particular single channel slice
    for h in range(new_h):  
        # At a given height, iterate through all the widths
        for w in range(new_w):
           # voxel = x[:, :, 0+h*pool_h: pool_h+h*pool_h , 0+w*pool_w:pool_w+w*pool_w]  # keep the number of images, and number channels untouched
            voxel = x[:, :, 0+h*stride: pool_h+h*stride , 0+w*stride:pool_w+w*stride]  # keep the number of images, and number channels untouched

            r_voxel = voxel.reshape(N, C, pool_w*pool_h)   # reshape the voxel, putting a flat 2-D image into a vector, but preserve the dimention for all the images and channels.
            out[:,:, h, w] = np.max(r_voxel, axis=2)       # conduct max pooling for each voxel
            out_index[:,:, h, w] = np.argmax(r_voxel, axis=2) # save the output max index information
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, out_index)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (x, pool_param, out_index) = cache
    (N, C, H, W) = x.shape               # the input image dimension 
    (N, C, new_h, new_w) = dout.shape    # the output image dimension
    pool_w = pool_param['pool_width']    # max pooling parameters
    pool_h = pool_param['pool_height']
    stride = pool_param['stride']
    
    dx = np.zeros_like(x)                # initialize the space for saving dx

    #dx_voxel = np.zeros((N, C, pool_h, pool_w))   # when condcut aech max pooling, what is the size for each voxel
    #dx_voxel_slice = np.zeros(( pool_h, pool_w))   # when condcut aech max pooling, what is the size for each slice in each voxel

    # At a location in the new image
    for h in range(new_h):  # in the output image, each height
        for w in range(new_w): # in the output image, each width
            maxID = out_index[:,:, h, w]   # what is the max ID for each voxel

            inter_h = np.array( maxID//pool_w, dtype=int )  # within each image of each channel, what is the maxID corresponding to the height in a voxel
            inter_w = np.array( maxID%pool_h , dtype=int )
            
            # for a pixel across all the images and all the channels, what is deravatives project back into the voxel 
            # in the corresponding location in the original input
            dx_voxel_new = np.zeros((N, C, pool_h, pool_w))  # reset a new dx_voxel values with zeros
            

            # update the derivative for a dx_voxel
            for n in range(N): # each image
                for c in range(C): # each channel
                    dx_voxel_new[n, c, inter_h[n,c], inter_w[n,c]]  = 1*dout[n, c, h, w] # update this voxel for this given location
                    
            # put the deravative for each voxel back into the location of the original image in dx
            dx[:, :, 0+h*stride: pool_h+h*stride, 0+w*stride:pool_w+w*stride] += dx_voxel_new

            del dx_voxel_new  # clear that object after use, because for each dx_voxel at a new location, it will be re-declare it. 

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    mode     = bn_param['mode']
    eps      = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N,C,H,W = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype)) # if 'running_mean' already exist, return its value; otherwise, initialize it is value
    running_var  = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))


    if mode == 'train':
        myx = x.reshape(N*H*W, C)
        
        # ----------------- The same as Batch Normailzation ------------------#
        sample_mean = np.mean(myx, axis=0)   # size (c,), mean of the current minibatch of data, across all the H and W
        sample_var  = np.var(myx, axis=0)    # size (c,), varirance of the current minibatch of data, across all the H and W

        x_minus_mean = myx - sample_mean # size of (C, N*H*W)
        sqrt_var = np.sqrt(sample_var+eps)   # size (c,)

        div = x_minus_mean/sqrt_var   # size of (C, N*H*W) normalize the data       
        multi = div * gamma          # size of (C, N*H*W) normalize the data  
        
        myout = multi + beta # scale and shift parameter 
        # --------------------------------------------------------------------#

        out = myout.reshape(N,C,H,W)   # reshape the data format back into its original size

        # update the running mean and var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var  = momentum * running_var  + (1 - momentum) * sample_var
        
        
        cache = (sample_mean,  sample_var, 
                 x_minus_mean, sqrt_var, 
                 div, multi, gamma, beta, eps, x)  # will be used during back propagation

    elif mode == 'test':
        
        myx = x.reshape(N*H*W, C)
       
        x_minus_mean = myx - running_mean # size of (C, N*H*W)
        sqrt_var = np.sqrt(running_var+eps)   # size (c,)
        
        div = x_minus_mean/sqrt_var   # size of (C, N*H*W) normalize the data       
        multi = div * gamma          # size of (C, N*H*W) normalize the data  
        
        myout = multi + beta # scale and shift parameter 
        # --------------------------------------------------------------------#

        out = myout.reshape(N,C,H,W)   # reshape the data format back into its original size

  
        
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)    
    
    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var']  = running_var


    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = dout.shape
    total_N = N*H*W
    mydout = dout.reshape(total_N, C)   
    #-------------------------------------------------------------------------#
    #-------------- All of these process are identical compared to -----------#
    #-------------- Batch Normalization earlier ------------------------------#
    ones_matrx = np.ones_like(mydout)  # later will use during dirivative calculation 
    
    (sample_mean,  sample_var, x_minus_mean, sqrt_var, 
     div, multi, gamma, beta, eps, x) = cache 
    y = div
    
    # backpro y*gamma + beta = out
    dbeta   = np.sum(mydout, axis=0)    # size(C,)  
    dgamma  = np.sum(y*mydout, axis=0)  # size(C,)    

    # backpro bacth_norm()
    # hand derive the equation from the computational graph for Batch Normalization 
    big_part = x_minus_mean/total_N * np.sum( - x_minus_mean/(sqrt_var**2) * mydout, axis=0)   
    double   = (gamma/sqrt_var) *(mydout + big_part)   
    dmyx       = 1/total_N*ones_matrx * np.sum(-double, axis=0) + double
   
    dx = dmyx.reshape(N,C,H,W)  # reshape back to it is original size
    #-------------------------------------------------------------------------#

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, bn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical 
    to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #mode  = bn_param['mode']
    N,C,H,W = x.shape
    out = np.zeros_like(x)
    group_len    = int(C/G)  # each group length
    
    gamma0 = gamma.reshape(C,)
    beta0  = beta.reshape(C,)
    
  
        
    cache_list = []
   
    previous_len = 0        
    for group in range(G):
        x_group = x[:, previous_len:previous_len+group_len, :, :]
        
        #myx = x_group.reshape(N, group_len*H*W )
        
        mygamma = gamma0[previous_len:previous_len+group_len]    # also need to group gamma
        mybeta  = beta0[previous_len:previous_len+group_len]     # also need to group beta
            
        (myout, mycache) = spatial_batchnorm_forward(x_group, mygamma, mybeta, bn_param)
        
        out[:, previous_len:previous_len+group_len, :, :] = myout#.reshape((N, group_len, H, W ))
        cache_list.append(mycache)

    cache = (x, cache_list, G, group_len)


        
        
        
        
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache




def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (x, cache_list, G, group_len) = cache
    dx = np.zeros_like(x)
    N,C,H,W = x.shape
    dgamma = np.zeros( C,)
    dbeta = np.zeros( C,)
    
    previous_len = 0 
    for group in range(G):
        dout_group = dout[:, previous_len:previous_len+group_len, :, :]
        mycache = cache_list[group]
        
        mydx, mydgamma, mydbeta = spatial_batchnorm_backward(dout_group, mycache)
        
        dgamma[ previous_len:previous_len+group_len]    = mydgamma
        dbeta[ previous_len:previous_len+group_len]     = mydbeta
        dx[:,previous_len:previous_len+group_len,:,:]   = mydx

    dgamma =  dgamma.reshape((1,C,1,1))
    dbeta =  dbeta.reshape((1,C,1,1))
    pass



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta




def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)   # normalize the data by minusing the maximum number, for calculation stability later
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
