import tensorflow as tf

'''Tools

Contains helper functions

In this script:
-squash
'''

def squash(inputs, axis=-1):
    '''Squash Function

    Squashes vectors such that short vectors get shrunk to a length of
    almost zero and long vectors get shrunk to a length of slighly below 1

    Args:
        inputs (tensor): A tensor of input vectors to squash 
        axis (int): The axis along which to do the norm calculation. Default
            is the last axis
    '''
    caps_norm = tf.norm(input_capsules, axis=axis, keepdims=True)
    return tf.square(caps_norm)/(1 + tf.square(caps_norm)) * (inputs/caps_norm)

def arg_2_list(input_arg, n=2, fill='repeat'):
    '''Takes an argument and converts it into a list of length n

    Args:
        input_arg (int, list or tuple): The input argument to be transformed.
            Must be 1 dimensional, and either length 1 or n.
        n (int): The length of the list

    Returns:
        output (list): A list of length n
    '''

    assert fill == 'repeat' or 'ones', 'Was expecting fill to be either' \
        'repeat or ones'
    # Input arg is longer than desired output
    if tf.shape(input_arg) > n: 
        raise ValueError('Input argument longer than list length n')
    # Input arg is already list of length n
    elif tf.shape(input_arg) == n: 
        return list(input_arg)
    # Input arg is in list form but only has length 1, repeat value n times
    elif tf.shape(input_arg) == 1: # Input arg is a
        output = list()
        if fill == 'repeat':
            output[i] = input_arg[0] for i in range(n)
        else: # fill == ones
            output = 1 for i in range(n)
            output[0] = input_arg[0]
        return output
    elif type(input_arg) == int:
        output = list()
        if fill == 'repeat':
            output[i] = input_arg for i in range(n)
        else: # fill == ones
            output = 1 for i in range(n)
            output[0] = input_arg
        return output
    else:
        raise ValueError('Got unexpected input argument when trying to convert to list')


