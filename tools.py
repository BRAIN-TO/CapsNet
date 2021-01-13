import tensorflow as tf

'''Tools

Contains helper functions

In this script:
-squash
-arg_2_list
-get_weight_matrixg
'''

def squash(inputs, axis=-1):
    '''Squash Function

    Squashes vectors such that short vectors get shrunk to a length of
    almost zero and long vectors get shrunk to a length of slighly below 1

    Args:
        inputs (tensor): A tensor of input vectors to squash. Note if inputs,
            are capsules, must flatten capsule matrices into vectors first
        axis (int): The axis along which to do the norm calculation. Default
            is the last axis
    '''
    inputs_norm = tf.norm(inputs, axis=axis, keepdims=True)
    return tf.square(inputs_norm)/(1 + tf.square(inputs_norm)) * (inputs/inputs_norm)

def arg_2_list(input_arg, n=2, fill='repeat'):
    '''Takes an argument and converts it into a list of length n

    Args:
        input_arg (int, list or tuple): The input argument to be transformed.
            Must be 1 dimensional, and either length 1 or n.
        n (int): The length of the list

    Returns:
        output (list): A list of length n, If value in input list is x,
            then the output list is either [x, x, x, x, ...] or
            [x, 1, 1, 1, ...]
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
    elif tf.shape(input_arg) == 1: # Input arg is a a list or tuple of length 1
        output = list()
        if fill == 'repeat':
            output[i] = [input_arg[0] for i in range(n)]
        else: # fill == ones
            output = [1 for i in range(n)]
            output[0] = input_arg[0]
        return output
    elif type(input_arg) == int: # Input is an int
        output = list()
        if fill == 'repeat':
            output[i] = [input_arg for i in range(n)]
        else: # fill == ones
            output = [1 for i in range(n)]
            output[0] = input_arg
        return output
    else:
        raise ValueError('Got unexpected input argument when trying to convert to list')

def get_weight_matrix(input_caps_dim, output_caps_dim):
    '''Determines the shape of the weight matrix

    Given the desired input and output dimensions, determines the necessary
    shape for the weight matrix as well as whether or not the inputs or
    outputs need to be transposed

    Args:
        input_caps_dim (list or tuple): 2 values describing the shape of
            the input capsules
        output_caps_dim (list or tuple): 2 values describing the shape of
            the output capsules

    Returns:
        w_shape (list): The shape of the wieght matrix
        trans_input (bool): Whether or not the input capsules need to be
            transposed before the operation
        trans_output (bool): Whether or not the output capsules need to
            be transposed after the operation
    '''

    if input_caps_dim[0] == output_caps_dim[0]:
        w_shape = [input_caps_dim[1], output_caps_dim[1]]
        trans_input = False
        trans_output = False
    elif input_caps_dim[0] == output_caps_dim[1]:
        w_shape = [input_caps_dim[1], output_caps_dim[0]]
        trans_input = False
        trans_output = True
    elif input_caps_dim[1] == output_caps_dim[0]:
        w_shape = [input_caps_dim[0], output_caps_dim[1]]
        trans_input = True
        trans_output = False
    elif input_caps_dim[1] == output_caps_dim[1]:
        w_shape = [input_caps_dim[0], output_caps_dim[0]]
        trans_input = True
        trans_output = True
    else:
        # matmul: input_caps * weights = output_caps
        # matmul shapes: [k, n] * [n, c] -> [k, c]
        # Hence input and output caps must share one dimension
        raise ValueError('Input capsule_dim must share one dimension with output capsule_dim')

    return w_shape, trans_input, trans_output