from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    
    http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf
    
    refer to slide (Vanilla) Recurrent Neural Network
    
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #f_w = lambda Wx, Wh, prev_h, x:  np.tanh( prev_h.dot(Wh) + x.dot(Wx) )
    #next_h = f_w(Wx, Wh, prev_h, x)
    fw_out = prev_h.dot(Wh) + x.dot(Wx) + b    # fw function
    next_h = np.tanh(fw_out) 
    
    cache = (Wh, prev_h, Wx, x, next_h)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (Wh, prev_h, Wx, X, next_h) = cache 
    dfw_out = (1-next_h**2)*(dnext_h) # back propagate tanh()
    
    # back propagate fw function
    dprev_h = dfw_out.dot(Wh.T)
    dWh     = (prev_h.T).dot(dfw_out)
    dx      = dfw_out.dot(Wx.T)
    dWx     = (X.T).dot(dfw_out)
    db      = np.sum(1*dfw_out, axis=0)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cache_list=[]
    (N, T, D) = x.shape
    H = Wx.shape[1]
    
    h = np.zeros((N,T,H))
    
    for t in range(T):
        myx = np.squeeze(x[:,t,:])
        next_h, cache_single = rnn_step_forward(myx, h0, Wx, Wh, b)
        h0  = next_h     # the next_h calculateed here will become the prev_h for next calculation
        
        h[:,t,:] = next_h[:,:]        
        cache_list.append(cache_single)
        
    
    cache = {'cache_list': cache_list,
             'N': N, 'T': T, 'D': D, 'H': H,} 
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cache_list =  cache['cache_list']
    N =  cache['N']
    T =  cache['T']
    D =  cache['D']
    H =  cache['H']          
    
    dx  = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros((H,))

    #dnext_h = dh
    for t in range(T-1,-1, -1): # reverse the order from T-1 to 0
        
        cache_single   = cache_list[t]
       
        # first time in the loop is the last hidden layer
        if t == T-1:   dnext_h_single = np.squeeze(dh[:,t,:])  # get a new upstream gradient        
        # other iterations are non-last hidden layer
        elif t != T-1: dnext_h_single = np.squeeze(dh[:,t,:])+dprev_h_single  # get a new upstream gradient

        dx_single, dprev_h_single, dWx_single, dWh_single, db_single = rnn_step_backward(dnext_h_single, cache_single)
                
        
        dx[:,t,:]= dx_single        
        dWx += dWx_single 
        dWh += dWh_single
        db  += db_single
        
    # for vallina recurrent neural network, the architechture inputs h0, which is the original h.
    # after that each h was recurrenttly calculated using fw. So when backprograge the first h, 
    # that is the gradient for h0.
    dh0 = dprev_h_single

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N, T) = x.shape
    (V, D) = W.shape
   
    out   = W[x]   # size (N,T,D)
    cache = (N,T,V,D,x)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N,T,V,D,x) = cache
    dW = np.zeros((V,D))
    
    #for n in range(N):
    #    dout_n = dout[n,:,:]        # size of (T,D)
    #    x_n    = x[n]               # for each example, get the x[n], it has size T, the indicies in T belongs to set V
    #    np.add.at(dW, x_n, dout_n)  # given the indicies in x_n, add the values of dout_n at these given indicies

    # conduct this operation with one line of code, instead of a for loop
    np.add.at(dW, x, dout) 
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (N,H)=prev_h.shape
    # prev_h: (N,H). Wh: (H, 4H).     X:(N,D). Wx:(D,4H). b:(4H).
    
    # step 1
    activation_matx = prev_h.dot(Wh) + x.dot(Wx) + b  
    
    # step 2
    # get each activation vector
    a_i = activation_matx[:,0:H] # size (N,H)
    a_f = activation_matx[:,H:2*H]
    a_o = activation_matx[:,2*H:3*H]
    a_g = activation_matx[:,3*H:4*H]
    
    # step 3
    i = sigmoid(a_i) # size (N,H)
    f = sigmoid(a_f)
    o = sigmoid(a_o)
    g = np.tanh(a_g)
    
    # step 4
    # f:(N,H). prev_c:(N,H).
    next_c = np.multiply(f, prev_c) + np.multiply(i,g)  # size (N,H)
    
    # step 5 
    last_tanh = np.tanh(next_c)  # size (N,H)
    
    # step 6
    next_h = np.multiply(o, last_tanh)  # size (N,H)

    pass

    cache = (Wh,prev_h, x, Wx, b, activation_matx, prev_c, i, g, f, o, last_tanh)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
     
    (Wh,prev_h, x, Wx, b, activation_matx, prev_c, i, g, f, o, last_tanh) = cache
    (N,H)=prev_h.shape
    # backpro  step 6
        # because all the functions here are element wise operation, so use '*', in python means elementwise operation,
        #  ... instead of .dot operation, which is matrix multiplication in python
    do         = last_tanh * dnext_h   # elementwise multiplication, same as np.multiply
    dlast_tanh = o * dnext_h
    # backpro  step 5 
    # here, first part is from the internal backpro of the computational graph,
    #   the second part dnext_c is from upstream of the network, which is given as input to this function. 
    dnext_c_in = (1-last_tanh**2)*dlast_tanh  +  dnext_c
    
    # backpro step 4
    # the branch without prev_c
        # because all the functions here are element wise operation, so use '*', in python means elementwise operation,
        #  ... instead of .dot operation, which is matrix multiplication in python
    di = g* dnext_c_in
    dg = i* dnext_c_in 
    # the branch with prev_c
    df      = prev_c*dnext_c_in
    dprev_c = f*dnext_c_in
    
    # backpro step 3
        # because all the functions here are element wise operation, so use '*', in python means elementwise operation,
        #  ... instead of .dot operation, which is matrix multiplication in python
    da_i = (i*(1-i)) *di  # sigmoid()
    da_g = (1-g**2)  *dg  # tanh()
    da_f = (f*(1-f)) *df  # sigmoid()
    da_o = (o*(1-o)) *do  # sigmoid()
    
    # backpro step 2
    dactivation_matx = np.zeros_like(activation_matx)  # size(N, 4H)
    dactivation_matx[:,0:H]     = da_i
    dactivation_matx[:,H:2*H]   = da_f
    dactivation_matx[:,2*H:3*H] = da_o
    dactivation_matx[:,3*H:4*H] = da_g
    
    # backpro step 1
    dprev_h = dactivation_matx.dot(Wh.T)       # (N, H)
    dWh     = (prev_h.T).dot(dactivation_matx) # (H, 4H)
    dx      = dactivation_matx.dot(Wx.T)       # (N, D) 
    dWx     = (x.T).dot( dactivation_matx)     # (D, 4H)
    db = np.sum(dactivation_matx, axis=0)      # size(4H,)
    
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cache_list = []
    prev_h     = h0   # initialize the first prev_h
    (N, T, D)  = x.shape
    H = Wh.shape[0]
    prev_c = np.zeros((N, H))   # initialize the prev_c for the first input hidden layer
    h = np.zeros((N, T, H))
    # at each individual time point, the size of myx is (N, D)
    for t in range(T):
        myx = np.squeeze(x[:,t,:])
        next_h, next_c, cache_single = lstm_step_forward(myx, prev_h, prev_c, Wx, Wh, b)

        prev_h = next_h   # the output of the next_h right now is the input of prev_h for next lstm hidden layer
        prev_c = next_c
        
        h[:,t,:] = next_h[:,:]   # save current hidden states
        cache_list.append(cache_single)
        
        
    cache = {'cache_list': cache_list,
             'N': N, 'T': T, 'D': D, 'H': H,}     
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx:  Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db:  Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    cache_list =  cache['cache_list']
    N =  cache['N']
    T =  cache['T']
    D =  cache['D']
    H =  cache['H']    
    
    dx  = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,4*H))
    dWh = np.zeros((H,4*H))
    db = np.zeros((4*H,))
    
    dnext_c_single = np.zeros((N, H))

    for t in range(T-1,-1, -1): # reverse the order from T-1 to 0
        
        cache_single   = cache_list[t]
        
        # first time in the loop is the last hidden layer
        if t == T-1:   dnext_h_single = np.squeeze(dh[:,t,:])  # get a new upstream gradient        
        # other iterations are non-last hidden layer
        elif t != T-1: dnext_h_single = np.squeeze(dh[:,t,:])+dprev_h_single  # get a new upstream gradient
        dx_single, dprev_h_single, dprev_c_single, dWx_single, dWh_single, db_single = lstm_step_backward(dnext_h_single, dnext_c_single, cache_single)
        
    
        dnext_c_single = dprev_c_single
                
        dx[:,t,:]= dx_single
        dnext_c_single
        dWx += dWx_single 
        dWh += dWh_single
        db  += db_single
      
    # for recurrent neural network, the architechture inputs h0, which is the original h.
    # after that each h was recurrenttly calculated. So when backprograge the first h, 
    # that is the gradient for h0.
    dh0 = dprev_h_single
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V   = x.shape

    x_flat    = x.reshape(N * T, V)
    y_flat    = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs  = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss   = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
