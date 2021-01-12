import tensorflow as tf
import tools

'''Routing

Contains functions for the routing algorithms for the various capsule 
layers

In This File:
-Dynamic Routing

TO-DO:
-Finish Dynamic Routing
-EM Routing
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
    # Flatten capsules into vectors
    input_shape = tf.shape(votes)
    vec_len = input_shape[-1] * input_shape[-2]
    u = tf.reshape(votes, shape=input_shape[0:-2] + [vec_len])

    # Only 1 prior per vote so get rid of caps dims and replace with dim 1
    b_shape = input_shape[0:-2] + [1]
    
    if priors == None: # Default, all priors zero, hence all coupling coeffs equal
        b = tf.constant(0, shape=b_shape, dtype=tf.float32)
    else: # Custom priors given
        if tf.shape(priors) == b_shape:
            b = priors
        elif tf.shape(priors) == b_shape[0:-1]: # [1] not added on
            b = tf.reshape(priors, shape=b_shape)
        else:
            raise ValueError('Priors for Dynamic Routing are wrong shape, should have one prior per vote')


    for i in range(num_iter):
        c = tf.nn.softmax(b) # Calculate coupling coefficients
        s = tf.reduce_sum(tf.multiply(c, u), axis=-2) # Take sume over all votes multiplied by coupling coefficients
        v = tools.squash(s, axis=-1) # Apply squash to get parent capsule predictions

        # Last step requires multiple lines of code for clarity
        a = tf.reduce_sum(u * tf.expand_dims(v, axis=-2), axis=-1) # Dot product of u and v is agreement
        a = tf.expand_dims(a, axis=-1) # Need to add empty dim so that it is same shape as b
        b = b + a # Update priors

    # reshape vectors back into two dimensional capsules.
    capsules = tf.reshape(v, shape=input_shape[0:-3] + [input_shape[-2], input_shape[-1]])

    return capsules