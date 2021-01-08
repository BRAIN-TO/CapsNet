import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tools

'''Layers

Contains the classes for all the different custom layer types used in this
project

In this file:
-PrimaryCaps2D
-ConvCaps2D
'''

class PrimaryCaps2D(layers.Layer):
    '''A Two Dimensional Primary Capsule Layer Class.

    The primary capsule layer is a convolutional capsule layer that uses
    a linear transformation to transform the outputs from a previous
    layer (usually a regular convolutional layer) into capsules. Not to
    be confused with ConvCaps2d layer. PrimaryCaps2d does not use any kind
    of routing to generate the capsules.
    
    Note that attributes listed below are only attributes that are defined
    internally and not defined by arguments to the init function.

    Attributes:
        kernel_dim (tuple): A tuple of 2 values defining the dimensions of
            the 2d convolution kernel
        strides (tuple): A tuple of 2 values defining the stride of convolution
            along each dimension
        built (boolean): Whether or not the layers build method has been run
        kernel (tensor): The tensor kernel weights for this layer. Trainable
        b (tensor): The tensor of biases for this layer. Trainable
    '''
    def __init__(self, num_channels, kernel_size, capsule_dim, stride=1, padding='same', activation='sigmoid', name=None, kernel_initializer='he_normal' **kwargs):
        '''A Two Dimensional Primary Capsule Layer 
        
        A convolutional capsule layer that converts activations from a 
        convolutional layer into capsules using a linear transformation

        Args:
            num_channels (int): The number channels in the capsule layer.
                Sort of like the number of filters for a convolutional layer
            kernel_size(list or int): An integer or tuple list of 2 integers,
                specifying the height and width of the 2d convolution 
                window.
            capsule_dim (int or list): The dimensionality of capsules in
                this layer. Capsules are vectors and hence capsule_dim
                is the number of values in the vector. This does not 
                include the capsule activation, which will be added. Will 
                also accept a list of integers describing the dimensions 
                of a tuple or pose matrix, which will in turn be flattened 
                into a vector during computation.
            stride (int or list): A list of two integers
                that specify the stride of convolution along the height
                and width respectively. Can be a single integer specifying
                the same value for both spatial dimensions
            padding (str): one of 'valid' or 'same', case sensitive.
                valid means no padding, same means even padding along
                both dimensions such that the output of the convolution
                is the same shape as the input
            activation (str): The method to use when calculating the capsule
                activations (probability of entity existence). Either
                'squash' or 'sigmoid'. The former uses the squash function
                on the capsule poses and then takes the length of the resultant
                vector. The latter uses the sigmoid function on a weighted
                sum of the activations in the previous layer.
            name (str): A name for the layer
            kernel_initializer (str): A string identifier for one of the 
                keras initializers. See the following documentation.
                https://www.tensorflow.org/api_docs/python/tf/keras/initializers
            **kwargs: Arbitrary keyword arguments for keras.layers.Layer()   
        '''
        super(PrimaryCaps2D, self).__init__(**kwargs) # Call parent init

        # Check input arguments
        ######################################
        assert len(kernel_size) < 3, 'kernel_size argument is too long.' \
            'Was expecting an int or list of size 2'
        assert len(stride) < 3, 'stride argument is too long' \
            'Was expecting an int or list of size 2'
        assert activation == 'squash' or 'sigmoid', 'Was expecting activation' \
            'to be either squash or sigmoid'

        # Set class attributes
        ######################################
        self.num_channels = num_channels
        self.kernel_dim = kernel_size if len(kernel_size) > 1 else (kernel_size, kernel_size) # if given int convert to tuple
        self.capsule_dim = capsule_dim if len(capsule_dim) == 1 else np.prod(capsule_dim) # Flatten if 2D
        self.strides = strides if len(strides) > 1 else (strides, strides) # if given int convert to tuple
        self.padding = padding
        self.activation = activation
        self.name = 'primary_caps2d' if name is None else name
        self.kernel_initializer = keras.initializers.get(weight_initializer)
        self.built = False   
    
    def build(self, input_shape):
        '''Builds the primary capsule layer.

        Args:
            input_shape (list or tuple): The input shape for the layer. 
                A list of integers that should be in the form:

                [batch_size, im_height, im_width, filters]

                where the last dimension is usually the number of filters
                for a convolutional layer
        '''
        # Define Trainable Variables
        #######################################

        # Essentially the same as the weights for a 2d convolutional filter
        # with num_channels*capsule_dim number of filters.
        # kernel weights used to calculate the capsule poses
        self.kernel = self.add_weight(
            name='kernel',
            shape=[self.kernel_dim[0], self.kernel_dim[1], input_shape[-1], self.num_channels*self.capsule_dim],
            initializer=self.kernel_initializer,
            trainable=True
        )
        # self.kernel shape: [k, k, input_filters, num_channels*capsule_dim]

        # 1 bias value for each output 'filter'. Hence num_channels*capsule_dim biases
        # Right now I have it shaped the same way keras would shape it
        # for a conv2d layer. However segcaps shapes their bias differently
        # Bias used to calculate the capsule poses
        self.b = self.add_weight(
            name='b',
            shape=[self.num_channels*self.capsule_dim,],
            trainable=True
        )

        # If sigmoid method chosen for calculating capsule activation then
        # We must have an additional set of weights for calculating capsule activations
        if self.activation == 'sigmoid'
            self.activation_weights = self.add_weight(
                name='activation_weights',
                shape=[self.kernel_dim[0], self.kernel_dim[1], input_shape[-1], self.num_channels],
                initializer=self.kernel_initializer,
                trainable=True
            )

        self.built = True
        
    def call(self, inputs):
        '''Performs the forward propagation to calculate the activations

        Args:
            inputs (tuple): The input activations to the layer. Should
                be of shape [batch_size, im_height, im_width, filters]

        Returns:
            capsules (tuple): A tuple containing the pose matrices and 
                activations for all the capsules in the layer.
        '''
        # Calculate the the capsule poses
        ######################################

        # Use a linear convolution to calculate the stacked poses
        poses_stacked = keras.backend.conv2d(
            x=inputs, 
            kernel=self.kernel, 
            strides=self.strides
            padding=self.padding,
            data_format='channels_last'
        ) # -> [batch_size, im_height, im_width, ]

        #  Add bias to the poses
        poses_stacked = tf.add(poses_stacked, self.b)

        # Reshape the poses into capsule form
        pose_shape = tf.shape(poses_stacked) # Has shape [batch_size, im_height, im_width, filters]

        # reshape into: [batch_size, im_height, im_width, num_channels, capsule_dim]
        capsule_poses = tf.reshape(poses_stacked, (-1, pose_shape[1], pose_shape[2], self.num_channels, self.capsule_dim))

        # Caculate the capsule activations
        ##########################################

        if self.activation == 'sigmoid': # Use Sigmoid Method
            # A convolution operation applies weights and takes the sum
            conv = keras.backend.conv2d(
                x=inputs,
                kernel=self.activation_weights,
                strides=self.strides,
                padding=self.padding,
                data_format='channels_last'
            )

            # Apply sigmoid function
            capsule_activations = tf.sigmoid(conv)

        else: # Use squash method
            # Use squash function on capsule poses
            pose_squashed = tools.squash(capsule_poses)

            # Calculate length of each capsule vector
            capsule_activations = tf.norm(pose_squashed, axis=-1) 

        # capsule_activations shape:
        # [batch_size, im_height, im_width, num_channels]
        
        # Concat capsule poses and activations into one tensor
        capsule_activations = tf.expand_dims(capsule_activations, axis=-1) # Add dimension of 1 to end
        capsules = tf.concat([capsule_poses, capsule_activations])

        # capsules tensor shape:
        # [batch_size, im_height, im_width, num_channels, caps_dim + 1]

        return capsules

class ConvCaps2D(layers.Layer):
    '''A Two Dimensional Convolutional Capsule Layer

    A convolutional capsule layer that uses routing from a previous layer
    of capsules to generate its own capsules. The inputs to this layer must
    be in capsule format.

    Note thast attributes listed below are only attributes that are defined
    internally and not defined by arguments to the init function.

    Attributes:

    '''
    def __init__(self, num_channels, kernel_size, capsule_dim, routing='EM', strides=1, padding='same', name=None, kernel_initializer='he_normal' **kwargs):
        '''A Two Dimensional Convolutional Capsule Layer

        A convolutional capsule layer that uses routing to generate higher
        level capsules from a previous capsule layer.

        Args:
            num_channels (int): The number channels in the capsule layer.
                Sort of like the number of filters for a convolutional layer
            kernel_size(list or int): An integer or tuple list of 2 integers,
                specifying the height and width of the 2d convolution 
                window.
            capsule_dim (int or list): The dimensionality of capsules in
                this layer. Capsules are vectors and hence capsule_dim
                is the number of values in the vector. This does not 
                include the capsule activation as a parameter in the vector 
                as it is calculated from the vector itself. Will also accept
                a list of integers describing the dimensions of a tuple or 
                pose matrix, which will in turn be flattened into a vector
                during computation. Eg. capsdim = (4, 4) -> 16
            strides (int or list): A list of two integers
                that specify the stride of convolution along the height
                and width respectively. Can be a single integer specifying
                the same value for both spatial dimensions
            padding (str): one of 'valid' or 'same', case sensitive.
                valid means no padding, same means even padding along
                both dimensions such that the output of the convolution
                is the same shape as the input
            name (str): A name for the layer
            kernel_initializer (str): A string identifier for one of the 
                keras initializers. See the following documentation.
                https://www.tensorflow.org/api_docs/python/tf/keras/initializers
            **kwargs: Arbitrary keyword arguments for keras.layers.Layer()   
        '''
        super(ConvCaps2D, self).__init__(**kwargs) # Call parent init

        # Check input arguments
        ######################################
        assert len(kernel_size) < 3, 'kernel_size argument is too long.' \
            'Was expecting an int or list of size 2'
        assert len(stride) < 3, 'stride argument is too long' \
            'Was expecting an int or list of size 2'

        # Set class attributes
        ######################################
        self.kernel_dim = kernel_size if len(kernel_size) > 1 else (kernel_size, kernel_size) # if given int convert to tuple
        self.num_channels = num_channels
        self.capsule_dim = capsule_dim if len(capsule_dim) == 1 else np.prod(capsule_dim) # flatten if 2D
        self.strides = strides if len(strides) > 1 else (strides, strides) # if given int convert to tuple
        self.padding = padding
        self.name = 'conv_caps2d' if name is None else name
        self.kernel_initializer = keras.initializers.get(weight_initializer)
        self.built = False

    def build(self, input_shape):
        '''Builds the convolutional capsule layer
            
        Args:
            input_shape (tuple): The shape of the capsule inputs
                A list of integers that should be in the form of:

                [batch_size, im_height, im_width, num_input_channels, input_caps_dim + 1]

                where input_caps_dim is the dimensionality of the capsules
                in the previoues layer and input_channels is the number
                of channels in the previous capsule layer. Note that the +1
                in the last dimension is due to the activations of the
                capsules being concatenated to the capsule poses
        '''

        # Define Trainable Variables
        # kernel has shape [kernel_height, kernel_width, num_input_channels, num_channels*capsule_dim]
        self.kernel = self.add_weight( 
            name='kernel',
            shape=[self.kernel_dim[0], self.kernel_dim[1], input_shape[-2], self.num_channels*self.capsule_dim],
            initializer=self.kernel_initializer,
            trainable=True
        ) 

        self.b = self.add_weight(
            name='bias',
            shape=[self.num_channels, self.capsule_dim],
            initializer=self.kernel_initializer,
            trainable=True
        )

        self.built = True

    def _get_conv_blocks(input_poses, kernel, strides):
        '''Computes capsule blocks from input poses using kernel weights

        Applies the kernel mask to the input poses, but unlike a
        convolution does not take the sum of the weighted capsules.
        Instead returns the block of weighted sums each with shape:

            [kernel_size, kernel_size, num_channels, caps_dim]

        Args:
            input_poses (tensor): The poses of the input capsules with
                shape: 
                    [batch_size, im_height, im_width, num_channels, caps_dim]
            kernel (tensor): The weights to be applied, should have shape:
                [kernel_size, kernel_size, num_input_channels, num_channels*caps_dim]
            strides (list):  A list of two integers that specify 
                the stride of convolution along the height and width respectively.

        Returns:
            blocks (tuple): A tensor containing one block of weighted 
                input capsule poses for each output capsule
        '''

    def call(self, inputs):
        '''Uses routing to perform the forward propagation of this layer

        Args:
            inputs (tuple): The input capsules/layer activations to this
                layer. Should have shape:

                [batch_size, im_height, im_width, num_input_channels, input_caps_dim + 1]

        Returns:
            capsules (tuple): A tuple containing the pose matrices and 
                activations for all the capsules in the layer.
        '''
        # Unpack Data
        ############################

        input_caps_dim = tf.shape(inputs)[-1] - 1 # Subtract one since activations add one to last dimension

        # First seperate the input capsule activations from the poses
        input_poses, input_activations = tf.split(
            inputs, 
            num_or_size_splits=[input_caps_dim, 1]
            )
        
        input_activations = tf.squeeze(input_activations, axis=-1) # Get rid of extra dim
        # input poses shape [batch_size, im_height, im_width, num_channels, caps_dim]
        # Input activations shape [batch_size, im_height, im_width, num_channels]

        # Calculate Votes from Poses
        ##############################




        
        

