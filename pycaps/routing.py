import tensorflow as tf
import pycaps.tools as tools
import math

'''Routing

Contains functions for the routing algorithms for the various capsule 
layers

In This File:
-Dynamic Routing
-EM Routing

TO-DO:
-Explore Leaky Routing
'''

def dynamic_routing(votes, num_iter=3, priors=None):
    '''Dynamic Routing Algorithm

    Implements the dynamic routing algorithm from Sabour et al. in:
    Dynamic Routing Between Capsules, 2017. 

    Args:
        votes: The votes for each output capsule. Last two dimensions 
            should be capsule_dim (the dimensions of each vote). The third
            last dimensions should be the number of votes per output capsule
        num_iter: The number of iterations to run the routing algorithm
        priors: The log prior probabilities for coupling between capsules,
            If left as none, all priors will be set to zero which results
            in equal couplind coefficients. Should have same shape as votes
    '''
    input_shape = tf.shape(votes) # [batch] + spatial_dims + [num_votes_per_cap, caps_dim1, caps_dim2]
    u = votes # rename votes variable to match paper notation

    # Only 1 prior per vote so replace both capsdims with 1
    b_shape = tf.concat([input_shape[:-2], [1, 1]], axis=0)

    if priors == None: # Default, all priors zero, hence all coupling coeffs equal
        b = tf.fill(b_shape, value=0.0)
    else: # Custom priors given
        try:
            b = tf.reshape(priors, shape=b_shape)
        except ValueError:
            print('Was unable to cast given priors for dynamic routing to the correct shape. Should have shape: ', b_shape)

    def _body(i, b, v_array): # A single routing iteration
        c = tf.nn.softmax(b) # Calculate coupling coefficients
        s = tf.reduce_sum(tf.multiply(c, u), axis=-3) # Take sume over all votes multiplied by their coupling coefficients
        v = tools.squash(s, axis=[-2, -1]) # Apply squash to get parent capsule predictions
        v_array = v_array.write(i, v)

        # Last step requires multiple lines of code for clarity
        a = tf.reduce_sum(u * tf.expand_dims(v, axis=-3), axis=[-2, -1]) # Dot product of u and v is agreement
        a = tf.expand_dims(tf.expand_dims(a, axis=-1), axis=-1) # Need to add empty dims so that it is same shape as b
        b = b + a # Update priors with agreement
        return (i+1, b, v_array)

    i = tf.constant(0, dtype=tf.int32)
    v_array = tf.TensorArray(dtype=tf.float32, size=num_iter, clear_after_read=False)
    _, b, v_array = tf.while_loop(
        cond=lambda i, b, v_array: i <  num_iter,
        body=_body,
        loop_vars=[i, b, v_array],
        swap_memory=True
    )
    # return final parent capsule predictions after all iterations
    return v_array.read(num_iter - 1)

def em_routing(votes, activations, beta_a, beta_u, out_channels, iterations=3):
    
    '''The EM Routing algorithm from the 2018 paper:

    Matrix Capsules with EM Routing by Geoffrey Hinton

    Args:
        votes (tensor): The votes for each output capsule. Last two dimensions 
            should be capsule_dim (the dimensions of each vote). The third
            last dimensions should be the number of votes per output capsule.
            The fourth last dimension should be the number of output capsules
        activations (tensor): The capsule activations associated with 
            each vote.
        beta_a (tensor): A discriminatively learned variable used in the
            routing. Cost of activating a capsule. 
        beta_u (tensor): A discriminatively learned variable used in the
            routing. Cost of not activating a capsule
        out_channels: The number of output channels of the layer. Used to set the
            starting values for the routing assignment matrix. For densecaps this
            is the number of capsules
        iterations (int): The number of iterations to run the routing algorithm
    '''
    # Input Votes shape
    # [batch_size, im_h, im_w, num_channels, num_votes_per_caps] + caps_dim
    # or [bacth_size, num_capsules, num_votes_per_caps] + caps_dim (for densecaps)
    votes_shape = tf.shape(votes)

    # Add two extra dims to activations, one for num_output_caps and one for flattened caps_dim
    a = tf.expand_dims(tf.expand_dims(tf.expand_dims(activations, axis=-2), axis=-1), axis=-1, name='vote_activations')
    # a shape: [batch_size, im_h, im_w, 1, num_votes_per_cap, 1, 1] or [batch_size, 1, num_votes_per_cap, 1, 1]

    # r shape: [num_capsules/num_channels, num_votes_per_caps, 1, 1]. Starts as 1/num_out_channels or 1/num_capsules for densecaps layer
    #r = tf.constant(1.0/votes.shape[-4], shape=list(votes.shape[-4:-2]) + [1, 1], dtype=tf.float32)
    r = tf.fill([votes_shape[-4], votes_shape[-3]], float(1.0/out_channels))
    r = tf.expand_dims(tf.expand_dims(r, axis=-1), axis=-1)
    r = tf.cast(r, dtype=tf.float32, name='routing_assignment_matrix')

    # b shape to [1, 1, 1, num_channels, 1, 1, 1] or [1, num_capsules, 1, 1, 1] for dense layer
    beta_a = tf.expand_dims(tf.expand_dims(tf.expand_dims(beta_a, axis=-1), axis=-1), axis=-1, name='beta_a')
    beta_u = tf.expand_dims(tf.expand_dims(tf.expand_dims(beta_u, axis=-1), axis=-1), axis=-1, name='beta_u')

    # Set min and max values for inverse temperature. Hyperparams taken from Jonathon Hui Implementation
    it_min = 1.0
    it_max = min(iterations, 3.0)

    # Iterate through routing algorithm
    for i in range(iterations):
        it = it_min + (i / max(1.0, it_max - 1.0)) * (it_max - it_min) # Set temperature value
        caps_means, caps_vars, a_out = _m_step(a, r, votes, beta_a, beta_u, it) # M-Step
        r = _e_step(caps_means, caps_vars, a_out, votes) # E-Step

    # Get rid of extra dimensions
    capsule_poses = tf.squeeze(caps_means, axis=-3)
    capsule_activations = tf.squeeze(a_out, axis=[-1, -2, -3])

    return capsule_poses, capsule_activations
    
def _m_step(a, r, v, beta_a, beta_u, it, epsilon=1e-9):
    '''M Step for EM routing algorithm

    Calculates the output capsule predictions given the input capsules and
    the soft routing assignments

    Args:
        a: The activations for the votes
        r: the routing assignment matrix
        v: the flattened capsule votes
        beta_a: The cost of activating a capsule
        beta_v: The cost of not activating a capsule
        it: the inverse temperature

    Returns:
        mean: the capsule means, estimated output capsules
        var: the capsule variances
        a_out: the estimated output capsule activations
    '''
    # Multiply r values by vote activations
    r_a = r * a # [batch_size, im_h, im_w, num_channels, num_votes_per_cap, 1, 1]
    r_a_sum = tf.reduce_sum(r, axis=-3, keepdims=True) # Used multiple times

    # Get mean over the axis of num_votes_per_caps [batch_size, im_h, im_w, num_channels, 1, caps_dim[0], caps_dim[1]]
    mean = tf.reduce_sum(r_a * v, axis=-3, keepdims=True) / r_a_sum

    # Get variance over the axis of num_votes_per_caps, same shape as mean
    var = tf.reduce_sum(r_a * tf.square(v - mean), axis=-3, keepdims=True) / r_a_sum

    # Calculate cost
    # Normalize cost for numeric stability
    cost = (beta_u + tf.math.log(tf.sqrt(var) + epsilon)) * r_a_sum
    cost_sum = tf.reduce_sum(cost, axis=[-1, -2], keepdims=True) # sum of cost across each caps_dim
    cost_mean = tf.reduce_mean(cost_sum, axis=-4, keepdims=True) # Take mean cost across each output channel
    cost_std = tf.sqrt(
        tf.reduce_sum(tf.square(cost_sum - cost_mean), axis=-4, keepdims=True) / cost_sum.shape[-3]
        )
    cost_norm = (cost_mean - cost_sum) / (cost_std + epsilon)

    # Calculate output activations
    a_out = tf.sigmoid(it * (beta_a + cost_norm))

    # Currently outputting 1's on first iteration
    # a_out shape [batch_size, im_h, im_w, num_channels, 1, 1, 1]

    return mean, var, a_out

def _e_step(mean, var, a_out, v, epsilon=1e-9):
    '''The E step for the EM routing algorithm

    Adjust the routing assignments of input capsules to output capsules
    given the current output capsule predictions

    Args:
        mean: the capsule means from the m step
        var: the capsule variances from the m step
        a_out: The estimated capsule activations from the m step
        v: the flattened capsule votes

    Returns:
        r: the routing assignment matrix
    '''
    # Note the math here for implementing this algorithm uses the efficient
    # implementation described in the paper but not with the equations in the
    # algorithm. 

    # sum of normalized votes for each capsule dimension
    vote_norm = tf.reduce_sum(
        tf.square(mean -  v) / 2 * var, axis=[-1, -2], keepdims=True
    )
    #vote_norm shape [batch_size, im_h, im_w, num_chan, 1, 1, 1]

    log_var = tf.reduce_sum(tf.math.log(tf.sqrt(var) + epsilon), axis=[-1, -2], keepdims=True) # sum of log_var for each capsule dimension
    log_prob = (-1 * vote_norm) - log_var - (tf.math.log(2*math.pi) / 2) # log probability density
    log_activations = tf.math.log(a_out + epsilon) # shape [batch_size, im_h, im_w, channels, 1, 1, 1]
    # log_prob and activation shape [batch_size] + spatial_shape + [channels/num_caps, 1, 1, 1]

    r = tf.nn.softmax(
        log_activations + log_prob, axis=-4
    ) # take softmax along num_caps/channels

    return r
