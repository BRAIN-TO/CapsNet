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
        if fill == 'repeat':
            output = [input_arg[0] for i in range(n)]
        else: # fill == ones
            output = [1 for i in range(n)]
            output[0] = input_arg[0]
        return output
    elif type(input_arg) == int: # Input is an int
        if fill == 'repeat':
            output = [input_arg for i in range(n)]
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

def safe_norm(input, epsilon=1e-7, axis=-1):
    '''A safe norm function that prevents the value being absolute zero
    
    Note this function was borrowed from Parth Rajesh Dedhia's CapsNet
    implementation

    Args:
        input (tensor): The tensor of vectors to run the norm function on
        epsilon (float): A near zero value to add to the norm function
            to prevent absolute zero values
        axis (int): The axis of input upon which to apply to norm function
    '''

    out_ = tf.reduce_sum(tf.square(input), axis=axis, keepdims=True)
    return tf.sqrt(out_ + epsilon)

def mask_output_capsules(labels, capsule_poses, weighted=False):
    '''Masks the output capsules using a set of labels

    Masks the output capsules by multiplying them by the labels. If 
    weighted is false, will convert labels to a binary mask by making
    the max label 1 and all other labels 0

    labels (tensor): A tensor containing either the activations or the 
        ground truth labels to mask the output capsules by
    capsule_poses (tensor): A tensor containing the capsule poses
    weighted (boolean): If true converts labels to a binary mask
    '''

    if weighted:    
         mask = tf.expand_dims(labels, axis=-1) # shape: [batch_size, num_caps, 1]
    else: # Convert labels to binary labels
        # Note that if labels are already binary this should not affect outcome
        # Create Mask with Predicted Class
        maxes = tf.equal(tf.reduce_max(labels, axis=-1, keepdims=True), labels) # Get boolean tensor with True for prediction and False otherwise
        mask = tf.where(
            maxes, 
            tf.constant(1, dtype=tf.float32), # 1 for max activation
            tf.constant(0, dtype=tf.float32) # 0 other capsules
        )
        mask = tf.expand_dims(mask, axis=-1) # shape: [batch_size, num_caps, 1]

     # Apply mask to pose
    pose_shape = capsule_poses.shape # pose shape [batch_size, num_capsules] + caps_dim
    pose_flat = tf.reshape(capsule_poses, [-1, pose_shape[1], pose_shape[-2] * pose_shape[-1]]) # flatten pose matrices into vectors
    pose_masked = tf.multiply(pose_flat, mask) # shape [batch_size, num_capsules] + caps_dim

    return pose_masked

