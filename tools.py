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