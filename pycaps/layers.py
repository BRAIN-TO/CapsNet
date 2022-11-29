# Public API's
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import dtypes
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops.gen_array_ops import pad
# Custom Imports
import pycaps.tools as tools
import pycaps.routing as routing


'''Layers

Contains the classes for all the different custom layer types used in this
project

In this file:
-PrimaryCaps2D
-DenseCaps
-ConvCaps2D

TO-DO/Improvements:
-Use absl to check inputs as opposed to if statements
-Add discriminative learning for log priors in dynamic routing algorithm
-Potentially move the get_votes and get_pose_blocks methods into tools
    in order to make this file more readable
-Potentially switch return order of pose and activations

'''

class PrimaryCaps2D(layers.Layer):
    '''A Two Dimensional Primary Capsule Layer Class.

    The primary capsule layer is a convolutional capsule layer that uses
    a linear transformation to transform the outputs from a previous
    non-capsule layer into capsules.
    
    Args:
        num_channels (int): The number channels in the capsule layer.
            Sort of like the number of filters for a convolutional layer
        kernel_size(list or int): An integer or tuple/list of 2 integers,
            specifying the height and width of the 2d convolution 
            window.
        capsule_dim (int or list): The dimensionality of capsule
            'poses' in this layer. Does not include the capsule
            activation value. Capsules can be either vectors or matrices.
            If given an int will treat capsules as a vector, if given
            a list or tuple of two values, will treat the capsules
            as matrices.
        strides (int or list): A list of two integers
            that specify the stride of convolution along the height
            and width respectively. Can be a single integer specifying
            the same value for both spatial dimensions
        padding (str): one of 'valid' or 'same', case sensitive.
            valid means no padding, same means even padding along
            both dimensions such that the output of the convolution
            is the same shape as the input
        activation (str): The method to use when calculating the capsule
            activations (probability of entity existence). Either
            'squash', norm' or 'sigmoid'. norm uses the length of the
            vector as the activation. squash uses the squash function
            and then takes the length. sigmoid uses the sigmoid function 
            on the weighted sum of the capsule poses from the previous layer.
            Note that sigmoid should only be used if the following layer
            uses EM routing
        name (str): A name for the layer
        kernel_initializer (str): A string identifier for one of the 
            keras initializers. See the following documentation.
            https://www.tensorflow.org/api_docs/python/tf/keras/initializers
        kernel_regularizer: A tensorflow regularizer instance
        bias_regularizer: A tensorflow regularizer instance
        **kwargs: Arbitrary keyword arguments for keras.layers.Layer()  
    '''
    def __init__(self, num_channels, kernel_size, capsule_dim, strides=1, padding='valid', activation='sigmoid', kernel_initializer='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super(PrimaryCaps2D, self).__init__(**kwargs) # Call parent init

        # Check input arguments
        ######################################
        assert len(tf.shape(kernel_size)) < 3, 'kernel_size argument is too long.' \
            'Was expecting an int or list of size 2'
        assert len(tf.shape(strides)) < 3, 'stride argument is too long' \
            'Was expecting an int or list of size 2'
        assert activation == 'squash' or 'sigmoid' or 'norm', 'Got unexpected' \
            'activation type'
        assert len(tf.shape(capsule_dim)) < 3, 'Capsules can be either vectors or matrices,' \
            'was expecting a capsule_dim no longer than 2 dimensions'

        # Set class attributes
        ######################################
        self.num_channels = num_channels
        self.kernel_dim = tuple(tools.arg_2_list(kernel_size, n=2)) # Convert to tuple of length 2
        self.capsule_dim = tools.arg_2_list(capsule_dim, n=2, fill='ones')
        self.strides = tuple(tools.arg_2_list(strides, n=2)) # Convert to tuple of length 2
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
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
        # with num_channels*np.prod(capsule_dim) number of filters.
        # kernel weights used to calculate the capsule poses
        self.kernel = self.add_weight(
            name='kernel',
            shape=[self.kernel_dim[0], self.kernel_dim[1], input_shape[-1], self.num_channels*np.prod(self.capsule_dim)],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        # self.kernel shape: [k, k, input_filters, num_channels*np.prod(capsule_dim)]

        # 1 bias value for each output 'filter'. Hence num_channels*np.prod(capsule_dim) biases
        # Right now I have it shaped the same way keras would shape it
        # for a conv2d layer. 
        self.b = self.add_weight(
            name='b',
            shape=[self.num_channels*np.prod(self.capsule_dim),],
            regularizer=self.bias_regularizer,
            trainable=True
        )

        # If sigmoid method chosen for calculating capsule activation then
        # We must have an additional set of weights for calculating capsule activations
        if self.activation == 'sigmoid':
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
            capsules (list): A list containing the capsule poses and 
                activations for all the capsules in the layer.
        '''
        # Calculate the the capsule poses
        ######################################

        # Use a linear convolution to calculate the stacked poses
        poses_stacked = keras.backend.conv2d(
            x=inputs, 
            kernel=self.kernel, 
            strides=self.strides,
            padding=self.padding,
            data_format='channels_last'
        ) # -> [batch_size, im_height, im_width, ]

        #  Add bias to the poses
        poses_stacked = tf.add(poses_stacked, self.b)

        # Reshape the poses into capsule form
        #pose_shape = poses_stacked.shape # Has shape [batch_size, im_height, im_width, filters]
        pose_shape = tf.shape(poses_stacked)

        # reshape into: [batch_size, im_height, im_width, num_channels, np.prod(self.capsule_dim)]
        capsule_poses = tf.reshape(poses_stacked, [-1, pose_shape[1], pose_shape[2], self.num_channels] + self.capsule_dim)

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
            ) # shape: [batch_size, im_height, im_width, num_channels]

            # Apply sigmoid function
            capsule_activations = tf.sigmoid(conv)

        else: # Use squash or norm method
            if self.activation == 'squash':
                # Use squash function on capsule poses
                capsule_poses = tools.squash(capsule_poses, axis=[-2, -1])

            # Calculate length of each capsule vector
            capsule_activations = tf.norm(capsule_poses, axis=[-2, -1]) 

        # capsule_activations shape:
        # [batch_size, im_height, im_width, num_channels]

        # capsules pose shape:
        # [batch_size, im_height, im_width, num_channels, caps_dim[0], caps_dim[1]]
        return [capsule_poses, capsule_activations]

    def get_config(self):
        '''Returns the layer configuration

        Allows the layer to be serializable

        Returns:
            config (dict): Dictionary containing th input arguments to
                the layer
        '''
        config = {
            'num_channels' : self.num_channels,
            'kernel_size' : self.kernel_dim,
            'capsule_dim': self.capsule_dim,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
        }
        base_config = super(PrimaryCaps2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def compute_output_shape(self, input_shape):
    #     # input shape is [batch_size, im_height, im_width, filters]
    #     # capsule_activations shape: [batch_size, im_height_new, im_width_new, num_channels]
    #     # capsules pose shape: [batch_size, im_height_new, im_width_new, num_channels, caps_dim[0], caps_dim[1]]
    #     h_new = conv_utils.conv_output_length(input_shape[1], self.kernel_dim[0], padding=self.padding, stride=self.strides[0])
    #     w_new = conv_utils.conv_output_length(input_shape[2], self.kernel_dim[1], padding=self.padding, stride=self.strides[1])
    #     a_shape = (input_shape[0], h_new, w_new, self.num_channels)
    #     p_shape = (input_shape[0], h_new, w_new, self.num_channels, self.capsule_dim[0], self.capsule_dim[1])
    #     return [p_shape, a_shape] # return shapes for both outputs

class ConvCaps2D(layers.Layer):
    '''A Two Dimensional Convolutional Capsule Layer

    A convolutional capsule layer that uses a kernel/mask of weights to
    calculate 'blocks' of votes using the capsules from a previous layer. The 
    kernel weights are applied using matrix multiplication as opposed to
    elementwise multiplication. Routing is then used to generate its capsules 
    rom each block of votes. The inputs to this layer must be in capsule format.
    '''
    def __init__(self, num_channels, kernel_size, capsule_dim, routing='EM', routing_iterations=3, strides=1, padding='valid', activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=None, **kwargs):
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
            routing (str): The kind of routing to use in this layer. Either
                'dynamic' or 'EM'. If EM, the capsule activations are calculated
                via the routing algorithm and the activation argument is ignored
            routing_iterations (int): The number of iterations to run the routing
                algorithm during the forward pass
            strides (int or list): A list of two integers
                that specify the stride of convolution along the height
                and width respectively. Can be a single integer specifying
                the same value for both spatial dimensions
            padding (str): one of 'valid' or 'same', case sensitive.
                valid means no padding, same means even padding along
                both dimensions such that the output of the convolution
                is the same shape as the input
            activation (str): The method to use when calculating the capsule
                activations (probability of entity existence). Either
                'squash', 'sigmoid' or 'norm'. norm uses the length of the
                vector as the activation. squash uses the squash function
                and then takes the length. sigmoid uses the sigmoid function 
                on the weighted sum of the capsule poses from the previous layer.
                Note that sigmoid should only be used if the following layer
                uses EM routing. If routing is 'EM' for this layer then this
                argument is ignored as EM routing generates its own activation
                values. Note that this is not the same as the activation function.
            kernel_initializer (str): A string identifier for one of the 
                keras initializers. See the following documentation.
                https://www.tensorflow.org/api_docs/python/tf/keras/initializers
            kernel_regularizer: A tensorflow regularizer instance
            **kwargs: Arbitrary keyword arguments for keras.layers.Layer()   
        '''
        super(ConvCaps2D, self).__init__(**kwargs) # Call parent init

        # Check input arguments (in hindsight should have done this with absl)
        ######################################
        assert len(tf.shape(kernel_size)) < 3, 'kernel_size argument is too long.' \
            ' Was expecting an int or list of size 2'
        assert len(tf.shape(strides)) < 3, 'stride argument is too long' \
            ' Was expecting an int or list of size 2'
        assert len(tf.shape(capsule_dim)) < 3, 'Capsules can be either vectors or matrices,' \
            ' was expecting a capsule_dim no longer than 2 dimensions'
        assert routing == 'EM' or 'dynamic', 'Was expecting routing' \
            ' argument to be either dynamic or EM'
        assert activation == 'squash' or 'sigmoid' or 'norm', 'Got unexpected' \
            ' activation type'

        # Set class attributes
        ######################################
        self.num_channels = num_channels
        self.kernel_dim = tuple(tools.arg_2_list(kernel_size, n=2)) # Convert to tuple of length 2
        self.capsule_dim = tools.arg_2_list(capsule_dim, n=2, fill='ones')
        self.strides = tuple(tools.arg_2_list(strides, n=2)) # Convert to tuple of length 2
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.built = False 
        self.routing = routing
        self.kernel_regularizer = kernel_regularizer
        self.routing_iter = routing_iterations

    def build(self, input_shapes):
        '''Builds the convolutional capsule layer
            
        Args:
            input_shape (list): A list containing the shapes for both
                the capsule poses and activations. Capsule pose shape
                should be in form:
                    [batch_size, im_height, im_width, num_input_channels] + input_caps_dim
                where input_caps_dim is the dimensionality of the capsules
                in the previoues layer 
        '''
        pose_shape = input_shapes[0] # [batch_size, im_h, im_w, num_input_chan] + caps_dim

        # Get shape values
        input_channels = pose_shape[-3]
        input_caps_dim = [pose_shape[-2], pose_shape[-1]]

        # Determine the shape of the weight matrices, and whether or not
        # The inputs or outputs need to be transposed to get desired shape.
        # This needs to be done since matrix multiplication is used as opposed to elementwise
        w_shape, self.trans_in, self.trans_out = tools.get_weight_matrix(input_caps_dim, self.capsule_dim)

        # Define Trainable Variables
        self.kernel = self.add_weight( 
            name='kernel',
            shape=[self.num_channels, self.kernel_dim[0], self.kernel_dim[1], input_channels] + w_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        ) 

        if self.routing == 'EM':
            # Discriminatively learned weights used in EM routing
            # One for each capsule type/channel
            self.beta_a = self.add_weight(
                name='beta_a',
                shape=[1, 1, 1, self.num_channels],
                initializer=keras.initializers.glorot_uniform()
            )
            self.beta_u = self.add_weight(
                name='beta_u',
                shape=[1, 1, 1, self.num_channels],
                initializer=keras.initializers.glorot_uniform()
            )

        self.built = True

    def _get_votes(self, pose_blocks):
        '''Calculates votes for routing
        
        Multiplies pose blocks by kernel weights to obtain votes for
        routing

        Args:
            pose_blocks (tuple): The poses of the input capsules in block
                form. Should be of shape:

                [batch_size, im_height, im_width, k_h, k_w, num_input_channels, input_caps_dim]

        Returns:
            votes (tuple): The votes for each output capsule.
        '''
        
        # Add empty dim to allow multiple output channels
        # [btch_s, im_h, im_w, k_h, k_w, in_chan] + in_caps_dim -> [btch_s, im_h, im_w, 1, k_h, k_w, in_chan] + in_caps_dim
        pose_blocks = tf.expand_dims(pose_blocks, axis=3)
        
        if self.trans_in == True:
            # Transpose the last two dimensions
            pose_blocks = tf.transpose(pose_blocks, [0, 1, 2, 3, 4, 5, 6, 8, 7])

        # kernel_shape [out_chan, k_h, k_w, in_chan, in_caps_dim]
        votes = tf.matmul(pose_blocks, self.kernel)

        if self.trans_out == True:
            # Transpose the last two dimensions
            votes = tf.transpose(votes, [0, 1, 2, 3, 4, 5, 6, 8, 7])

        # final votes shape
        # [batch_size, im_h, im_w, out_chan, k_h, k_w, in_chan] + out_caps_dim
        return votes

    def call(self, inputs):
        '''Uses routing to perform the forward propagation of this layer

        Args:
            inputs (list): A list containing the tuples for capsule poses
                and activations. Capsule poses should have shape:
                    [batch_size, im_height, im_width, num_input_channels] + input_caps_dim

        Returns:
            capsules (list): A list containing the capsule poses and 
                the capsule activations
        '''
        # Unpack Data
        ############################
        input_poses = inputs[0]
        input_activations = inputs[1]
        pose_shape = tf.shape(input_poses)
        input_caps_dim = [pose_shape[-2], pose_shape[-1]]

        # input poses is 6D tensor with shape [batch_size, input_height, input_width, num_input_channels, input_caps_dim[0], input_caps_dim[1]]
        # Input activations is 4D tensor with shape [batch_size, input_height, input_width, num_input_channels]

        # Calculate Votes from Poses
        ##############################
        # pose_flat is 4D tensor with shape [batch_size, input_height, input_width, num_input_channels*input_caps_dim[0]*input_caps_dim[1]]
        poses_flat = tf.reshape(input_poses, shape=[-1, pose_shape[1], pose_shape[2], pose_shape[3]*pose_shape[4]*pose_shape[5]])
        

        # Patches is 4D tensor with shape [batch_size, out_height, out_width, num_channels * input_caps_dim[0] * input_caps_dim[1] * k_h * k_w]
        patches = tf.image.extract_patches(
            poses_flat, 
            sizes=[1, self.kernel_dim[0], self.kernel_dim[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper()
        )

        
        patches_shape = tf.shape(patches)
        # blocks is 8D tensor with shape [batch_size, out_height, out_width, k_h, k_w, num_input_channels] + input_caps_dims
        blocks = tf.reshape(patches, shape=[-1, patches_shape[1], patches_shape[2], self.kernel_dim[0], self.kernel_dim[1], pose_shape[3], pose_shape[4], pose_shape[5]])

        votes = self._get_votes(blocks) # has shape [batch_size, height, width, out_chan, k_h, k_w, in_chan] + out_caps_dim

        # Get Image Shape Attributes
        ##############################
        shape = tf.shape(votes) # [batch_size, height, width, out_chan, k_h, k_w, in_chan] + out_caps_dim
        im_height = shape[1] # Output height
        im_width = shape[2] # Output width
        num_votes_per_capsule = shape[4]*shape[5]*shape[6] # k_h * k_w *  in_chan

        # Reshape votes, flattening k_h, k_w and in_chan into one dimension
        votes = tf.reshape(votes, [-1, im_height, im_width, self.num_channels, num_votes_per_capsule] + self.capsule_dim)

        # Routing
        ############################
        if self.routing == 'dynamic':
            # capsule_poses shape: [batch_size, im_height, im_width, num_out_channels] + capsule_dim
            capsule_poses = routing.dynamic_routing(votes, num_iter=self.routing_iter)

             # Calculate new capsule activations
            if self.activation == 'sigmoid':
                raise ValueError('Sigmoid activation not yet supported for ConvCaps2D')
            else:
                if self.activation == 'squash':
                    # Use squash function on capsule poses
                    capsule_poses = tools.squash(capsule_poses, axis=[-2, -1])

                # Calculate length of each capsule vector
                capsule_activations = tf.norm(capsule_poses, axis=[-2, -1]) 

        elif self.routing == 'EM':
            # vote_activations is 4D tensor with shape [batch_size, im_h, im_w, num_votes_per_capsule]
            vote_activations = tf.image.extract_patches(
                input_activations,
                sizes=[1, self.kernel_dim[0], self.kernel_dim[1], 1],
                strides=[1, self.strides[0], self.strides[1], 1],
                rates=[1, 1, 1, 1],
                padding=self.padding.upper()
                )

            capsule_poses, capsule_activations = routing.em_routing(votes, vote_activations, self.beta_a, self.beta_u, self.num_channels, self.routing_iter)

        else:
            raise ValueError('Was expecting either EM or dynamic for routing argument')

        return [capsule_poses, capsule_activations]

    def get_config(self):
        '''Returns the layer configuration

        Allows the layer to be serializable

        Returns:
            config (dict): Dictionary containing th input arguments to
                the layer
        '''
        config = {
            'num_channels' : self.num_channels,
            'kernel_size' : self.kernel_dim,
            'capsule_dim': self.capsule_dim,
            'routing': self.routing,
            'stride': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
        }
        base_config = super(ConvCaps2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 

class DenseCaps(layers.Layer):
    '''A Dense Layer For Capsules
    
    A fully connected capsule layer that uses routing to generate 
    higher level capsules from a previous capsule layer. Note that 
    inputs must be flattened to 1 spatial dimension. Also uses matrix multiplication
    as opposed to elementwise multiplication when applying weights.

    Args:
        num_channels (int): The number of capsules in the layer
        capsule_dim (int or list): The dimensionality of capsule
            'poses' in this layer. Does not include the capsule
            activation value. Capsules can be either vectors or matrices.
            If given an int will treat capsules as a row vector, if given
            a list or tuple of two values, will treat the capsules
            as matrices.
        routing (str): The kind of routing to use in this layer. Either
            'dynamic' or 'EM'. If EM, the capsule activations are calculated
            via the routing algorithm and the activation argument is ignored
        routing_iterations (int): The number of iterations to run the routing
            algorithm during the forward pass
        activation (str): The method to use when calculating the capsule
            activations (probability of entity existence). Either
            'squash', 'sigmoid' or 'norm'. norm uses the length of the
            vector as the activation. squash uses the squash function
            and then takes the length. sigmoid uses the sigmoid function 
            on the weighted sum of the capsule poses from the previous layer.
            Note that sigmoid should only be used if the following layer
            uses EM routing. If routing is 'EM' for this layer then this
            argument is ignored
        add_coordinates (bool): If true will add the scaled indices 
            of the spatial shape of the inputs to the values within the 
            capsule poses indicated by the pose_coords argument. The spatial
            dimensions are assumed to be the dimensions between the first 
            (batch_size) dimension and the third last (num_channels)
            dimension. Note it is not recommended to use this feature 
            when there are more than 3 spatial dimensions. Can not be used
            if previous layer is a dense layer
        pose_coords (list): A list of 2d coordinates (row, column) indicating 
            which value within the capsule pose matrices to add the coordinates
            from the previous layers spatial shape. The order of coordinates
            corresponds to the order of spatial dimensions in the input layer
        initializer (str): A string identifier for one of the 
                keras initializers. See the following documentation.
                https://www.tensorflow.org/api_docs/python/tf/keras/initializers
        regularizer: A tensorflow regularizer instance applied to layer weights
        **kwargs: Arbitrary keyword arguments for keras.layers.Layer()   
    '''
    def __init__(self, num_capsules, capsule_dim, routing='EM', routing_iterations=3, activation='squash', add_coordinates=False, pose_coords=None, initializer='random_normal', regularizer=None, **kwargs):
        super(DenseCaps, self).__init__(**kwargs)

        # Check inputs
        assert activation == 'squash' or 'sigmoid' or 'norm', 'Got unexpected' \
            ' activation type'
        assert routing == 'EM' or 'dynamic', 'Was expecting routing' \
            ' argument to be either dynamic or EM'
        assert len(tf.shape(capsule_dim)) < 3, 'Capsules can be either vectors or matrices,' \
            ' was expecting a capsule_dim no longer than 2 dimensions'
        assert type(num_capsules) == int, 'num_capsules must be an integer'

        self.num_capsules = num_capsules
        self.capsule_dim = tools.arg_2_list(capsule_dim, n=2, fill='ones') # Convert capsule dim in 2d
        self.routing = routing
        self.activation = activation
        self.built = False
        self.initializer = keras.initializers.get(initializer)
        self.add_coordinates = add_coordinates
        self.pose_coords = pose_coords
        self.regularizer = regularizer
        self.routing_iter = routing_iterations

    def build(self, input_shapes):
        '''Builds the dense capsule layer

        Args:
            input_shapes (tensor): A list containing the shapes of the 
                capsule poses and activations. 
        '''
        # pose_shape: [batch size] + spatial_shape + [num_channels] + input_caps_dim, if previous layer is convcaps or primary layer
        # pose_shape: [batch_size, num_capsules, input_caps_dim[0], input_caps_dim[1]], if previous layer is densecaps layer
        pose_shape = input_shapes[0] 

        # Get the shape of the weight matrices
        input_caps_dim = pose_shape[-2:]
        middle_dims = pose_shape[1:-2] # contains spatial_shape + num_channels or just num_capsules
        w_shape, self.trans_in, self.trans_out = tools.get_weight_matrix(input_caps_dim, self.capsule_dim)

        # Create the weight matrices
        self.w = self.add_weight(
            name='w',
            shape=[self.num_capsules] + middle_dims + [w_shape[0], w_shape[1]],
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True
        )

        if self.routing == 'EM':
            self.beta_a = self.add_weight(
                name='beta_a',
                shape=[1, self.num_capsules],
                initializer=keras.initializers.glorot_uniform()
            )
            self.beta_u = self.add_weight(
                name='beta_u',
                shape=[1, self.num_capsules],
                initializer=keras.initializers.glorot_uniform()
            )
        #elif self.activation == 'sigmoid':


        self.built = True

    def call(self, inputs):
        '''Uses routing to generate parent capsules

        Multiplies inputs by wieghts to get votes and then uses routing
        to transform votes into parent capsules
        
        Args:
            inputs (list): A list containing both the input capsules poses
                activations.
        '''

        assert type(inputs) == list, 'Was expecting layer inputs to be a' \
            ' list containing a tensor for poses and a tensor for activations'

        # [batch_size, num_input_caps] + input_caps_dim
        # or [batch_size] + spatial_shape + [num_input_channels] + input_caps_dim
        input_poses = inputs[0] 
        input_activations = inputs[1]
        
        # Calculate Votes
        ##########################

        # Add dim for output channels
        input_poses = tf.expand_dims(input_poses, axis=1) # shape: [batch_size, 1] + middle_shape + input_caps_dim

        if self.trans_in == True: # Transpose last two dimensions (in_caps_dim)
            input_poses = tf.linalg.matrix_transpose(input_poses)

        # Multiply input poses by the weights
        votes = tf.matmul(input_poses, self.w) 

        if self.trans_out == True: # Transpose last two dimensions (out_caps_dim)
            votes = tf.linalg.matrix_transpose(votes)

        if self.add_coordinates:
            assert self.pose_coords is not None, 'add_coordinates is true but was' \
                ' not provided a list of pose values in pose_coords to add the coordinates to'
            
            votes = tools.add_coordinates(votes, self.pose_coords)

        # votes shape [batch_size, num_capsules] + spatial_shape + [num_input_channels] + out_caps_dim
        #votes_shape = votes.shape
        votes_shape = tf.shape(votes)

        # Flatten votes
        num_votes_per_cap = tf.reduce_prod(votes_shape[2:-2]) # takes prod of spatial shape and num_input_channels
        votes =  tf.reshape(votes, shape=[-1, self.num_capsules, num_votes_per_cap, self.capsule_dim[0], self.capsule_dim[1]])

        # Routing
        #######################
        if self.routing == 'dynamic':
            capsule_poses = routing.dynamic_routing(votes, num_iter=self.routing_iter)

            # Calculate new capsule activations
            if self.activation == 'sigmoid':
                raise ValueError('Sigmoid activation not yet supported for DenseCaps')
            else:
                if self.activation == 'squash':
                    # Use squash function on capsule poses
                    capsule_poses = tools.squash(capsule_poses, axis=[-2, -1])

                # Calculate length of each capsule vector
                capsule_activations = tf.norm(capsule_poses, axis=[-2, -1])

        elif self.routing == 'EM':
            # flatten activations
            a = tf.reshape(input_activations, shape=[-1, num_votes_per_cap])
            capsule_poses, capsule_activations = routing.em_routing(votes, a, self.beta_a, self.beta_u, self.num_capsules, self.routing_iter)
        else:
            raise ValueError('Was expecting either EM or dynamic for routing argument')
            
        # pose shape [batch_size, num_capsules] + output_caps_dim
        # activation shape [batch_size, num_capsules]

        return [capsule_poses, capsule_activations]

    def get_config(self):
        '''Returns the layer configuration

        Allows the layer to be serializable

        Returns:
            config (dict): Dictionary containing th input arguments to
                the layer
        '''
        config = {
            'num_capsules': self.num_capsules,
            'capsule_dim': self.capsule_dim,
            'routing': self.routing,
            'activation': self.activation,
            'initializer': self.initializer
        }
        base_config = super(DenseCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))