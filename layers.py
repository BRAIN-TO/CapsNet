# Public API's
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# Custom Imports
import tools
import routing


'''Layers

Contains the classes for all the different custom layer types used in this
project

In this file:
-PrimaryCaps2D
-DenseCaps
-ConvCaps2D

TO-DO:
-Add different activation methods for convcaps and densecaps
-Change dense layer to work with having previous layer also being dense by
adding a flatten layer
-Add discriminative learning for log priors in dynamic routing algorithm
-Support the use of a bias in conv caps
-Potentially move the get_votes and get_pose_blocks methods into tools
    in order to make this file more readable
-Potentially switch return order of pose and activations
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
    '''
    def __init__(self, num_channels, kernel_size, capsule_dim, strides=1, padding='valid', activation='sigmoid', kernel_initializer='glorot_uniform', **kwargs):
        '''A Two Dimensional Primary Capsule Layer 
        
        A convolutional capsule layer that converts activations from a 
        convolutional layer into capsules using a linear transformation

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
                'squash', 'sigmoid' or 'norm'. norm uses the length of the
                vector as the activation. squash uses the squash function
                and then takes the length. sigmoid uses the sigmoid function 
                on the weighted sum of the capsule poses from the previous layer
            name (str): A name for the layer
            kernel_initializer (str): A string identifier for one of the 
                keras initializers. See the following documentation.
                https://www.tensorflow.org/api_docs/python/tf/keras/initializers
            **kwargs: Arbitrary keyword arguments for keras.layers.Layer()   
        '''
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
            trainable=True
        )
        # self.kernel shape: [k, k, input_filters, num_channels*np.prod(capsule_dim)]

        # 1 bias value for each output 'filter'. Hence num_channels*np.prod(capsule_dim) biases
        # Right now I have it shaped the same way keras would shape it
        # for a conv2d layer. 
        self.b = self.add_weight(
            name='b',
            shape=[self.num_channels*np.prod(self.capsule_dim),],
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
        pose_shape = poses_stacked.shape # Has shape [batch_size, im_height, im_width, filters]

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
            # Flatten capsule poses into vectors
            shape = capsule_poses.shape
            vectors = tf.reshape(capsule_poses, shape=[-1, shape[1], shape[2], shape[3], shape[4]*shape[5]])

            if self.activation == 'squash':
                # Use squash function on capsule poses
                vectors = tools.squash(vectors)

            # Calculate length of each capsule vector
            capsule_activations = tf.norm(vectors, axis=-1) 

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

class DenseCaps(layers.Layer):
    '''A Dense Layer For Capsules

    Uses a set of weights to calculate votes from input capsules. Similar
    to a fully connected layer, each input capsule is connected (casts a vote)
    to each output/parent capsule. Routing is then used to generate the 
    output capsules from the votes.
    '''
    def __init__(self, num_capsules, capsule_dim, routing='EM', activation='squash', initializer='random_normal', **kwargs):
        '''A Dense Layer for Capsules

        A fully connected capsule layer that uses routing to generate 
        higher level capsules from a previous capsule layer.

        Args:
            num_channels (int): The number of capsules in the layer
            capsule_dim (int or list): The dimensionality of capsule
                'poses' in this layer. Does not include the capsule
                activation value. Capsules can be either vectors or matrices.
                If given an int will treat capsules as a vector, if given
                a list or tuple of two values, will treat the capsules
                as matrices.
            routing (str): The kind of routing to use in this layer. Either
                'dynamic' or 'EM'
            activation (str): The method to use when calculating the capsule
                activations (probability of entity existence). Either
                'squash', 'sigmoid' or 'norm'. norm uses the length of the
                vector as the activation. squash uses the squash function
                and then takes the length. sigmoid uses the sigmoid function 
                on the weighted sum of the capsule poses from the previous layer
            name (str): A name for the layer
            **kwargs: Arbitrary keyword arguments for keras.layers.Layer()   
        '''

        super(DenseCaps, self).__init__(**kwargs)

        # Check inputs
        assert activation == 'squash' or 'sigmoid' or 'norm', 'Got unexpected' \
            'activation type'
        assert routing == 'EM' or 'dynamic', 'Was expecting routing' \
            'argument to be either dynamic or EM'
        assert len(tf.shape(capsule_dim)) < 3, 'Capsules can be either vectors or matrices,' \
            'was expecting a capsule_dim no longer than 2 dimensions'
        assert type(num_capsules) == int, 'num_capsules must be an integer'

        self.num_capsules = num_capsules
        self.capsule_dim = tools.arg_2_list(capsule_dim, n=2, fill='ones') # Convert capsule dim in 2d
        self.routing = routing
        self.activation = activation
        self.built = False
        self.initializer = keras.initializers.get(initializer)

    def build(self, input_shapes):
        '''Builds the dense capsule layer

        Args:
            input_shapes (tensor): A list containing the shapes of the 
                capsule poses and activations. For now the inputs should be
                from a convolutional capsule layer hence should start with shape
                    [batch_size, im_height, im_width, input_channels] + input_caps_dim
        '''
        pose_shape = input_shapes[0]

        input_caps_dim = pose_shape[-2:]
        w_shape, self.trans_in, self.trans_out = tools.get_weight_matrix(input_caps_dim, self.capsule_dim)

        self.w = self.add_weight(
            name='w',
            shape=[self.num_capsules, pose_shape[1], pose_shape[2], pose_shape[3], w_shape[0], w_shape[1]],
            initializer=self.initializer,
            trainable=True
        )

        # Potentially add bias later

        self.built = True

    def call(self, inputs):
        '''Uses routing to generate parent capsules

        Multiplies inputs by wieghts to get votes and then uses routing
        to transform votes into parent capsules
        
        Args:
            inputs (tensor): The input capsules to the layer. For now the
                inputs should be from a convolutional capsule layer and
                hence should have shape
                    [batch_size, im_height, im_width, input_channels] + input_caps_dim
        '''

        assert type(inputs) == list, 'Was expecting layer inputs to be a' \
            'list containing a tensor for poses and a tensor for activations'

        input_poses = inputs[0]
        input_activations = inputs[1]
        
        # Calculate Votes
        ##########################

        # Add dim for output channels
        input_poses = tf.expand_dims(input_poses, axis=1) # shape: [batch_size, 1, im_h, im_w, in_chan] + in_caps_dim

        if self.trans_in == True: # Transpose last two dimensions (in_caps_dim)
            input_poses = tf.transpose(input_poses, [0, 1, 2, 3, 4, 6, 5])

        votes = tf.matmul(input_poses, self.w) 

        if self.trans_out == True: # Transpose last two dimensions (out_caps_dim)
            votes = tf.transpose(votes, [0, 1, 2, 3, 4, 6, 5])

        # votes shape [batch_size, num_capsules, im_h, im_w, in_chan] + out_caps_dim

        # Routing
        #######################
        votes_shape = votes.shape
        num_votes_per_caps = votes_shape[2]*votes_shape[3]*votes_shape[4] # im_h * im_w * in_chan
        votes = tf.reshape(votes, shape=[-1, self.num_capsules, num_votes_per_caps, self.capsule_dim[0], self.capsule_dim[1]])
        # votes new shape [batch_size, num_capsules, im_h * im_w * in_chan] + out_caps_dim

        if self.routing == 'dynamic':
            capsule_poses = routing.dynamic_routing(votes)
        elif self.routing == 'EM':
            raise ValueError('EM Routing Not Yet Implemented. Please select dynamic instead')
        else:
            raise ValueError('Was expecting either EM or dynamic for routing argument')

        # Calculate new capsule activations
        if self.activation == 'activation':
            raise ValueError('Sigmoid activation not yet supported for DenseCaps')
        else:
            # Flatten capsule poses into vectors
            shape = capsule_poses.shape
            vectors = tf.reshape(capsule_poses, shape=[-1, shape[1], shape[2]*shape[3]])
            if self.activation == 'squash':
                # Use squash function on capsule poses
                vectors = tools.squash(vectors)

            # Calculate length of each capsule vector
            capsule_activations = tf.norm(vectors, axis=-1)
            

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


class ConvCaps2D(layers.Layer):
    '''A Two Dimensional Convolutional Capsule Layer

    A convolutional capsule layer that uses a kernel/mask of weights to
    calculate 'blocks' of votes using the capsules from a previous layer.
    It then uses routing to generate its own capsules from these votes. 
    The inputs to this layer must be in capsule format.
    '''
    def __init__(self, num_channels, kernel_size, capsule_dim, routing='EM', strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', **kwargs):
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
                'dynamic' or 'EM'
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
        super(ConvCaps2D, self).__init__(**kwargs) # Call parent init

        # Check input arguments
        ######################################
        assert tf.shape(kernel) < 3, 'kernel_size argument is too long.' \
            'Was expecting an int or list of size 2'
        assert tf.shape(stride) < 3, 'stride argument is too long' \
            'Was expecting an int or list of size 2'
        assert activation == 'squash' or 'sigmoid', 'Was expecting activation' \
            'to be either squash or sigmoid'
        assert tf.shape(capsule_dim) < 3, 'Capsules can be either vectors or matrices,' \
            'was expecting a capsule_dim no longer than 2 dimensions'
        assert routing == 'EM' or 'dynamic', 'Was expecting routing' \
            'argument to be either dynamic or EM'
        assert activation == 'squash' or 'sigmoid' or 'norm', 'Got unexpected' \
            'activation type'

        # Set class attributes
        ######################################
        self.num_channels = num_channels
        self.kernel_dim = tuple(tools.arg_2_list(kernel_size, n=2)) # Convert to tuple of length 2
        self.capsule_dim = tools.arg_2_list(capsule_dim, n=2, fill='ones')
        self.strides = tuple(tools.arg_2_list(stride, n=2)) # Convert to tuple of length 2
        self.padding = padding
        self.activation = activation
        self.name = 'primary_caps2d' if name is None else name
        self.kernel_initializer = keras.initializers.get(weight_initializer)
        self.built = False 
        self.routing = routing

    def build(self, input_shape):
        '''Builds the convolutional capsule layer
            
        Args:
            input_shape (list): A list containing the shapes for both
                the capsule poses and activations. Capsule pose shape
                should be in form:
                    [batch_size, im_height, im_width, num_input_channels] + input_caps_dim
                where input_caps_dim is the dimensionality of the capsules
                in the previoues layer 
        '''
        pose_shape = input_shape[0]

        # Get shape values
        input_channels = pose_shape[-3]
        input_caps_dim = [pose_shape[-2], pose_shape[-1]]

        # Determine the shape of the weight matrices, and whether or not
        # The inputs or outputs need to be transposed to get desired shape
        w_shape, self.trans_in, self.trans_out = tools.get_weight_matrix(input_caps_dim, self.capsule_dim)

        # Define Trainable Variables
        self.kernel = self.add_weight( 
            name='kernel',
            shape=[self.num_channels, self.kernel_dim[0], self.kernel_dim[1], input_channels] + w_shape,
            initializer=self.kernel_initializer,
            trainable=True
        ) 

        # Potentially add a bias term later

        self.built = True

    def _get_pose_blocks(input_poses):
        '''Transforms input capsule poses into blocks

        Uses the kernel size to transform the input poses into the block
        form required to multiply them by the kernel weights.
            [kernel_size, kernel_size, num_input_channels, input_caps_dim[0], input_caps_dim[1]]

        Args:
            input_poses (tensor): The poses of the input capsules with
                shape: 
                    [batch_size, im_height, im_width, num_input_channels] + input_caps_dims

        Returns:
            blocks (tuple): A tensor containing the pose blocks
        '''

        input_shape = tf.shape(input_poses)

        # Get number of steps for both height and width
        h_steps = int((input_shape[1] - self.kernel_dim[0] + 1)/self.strides[0])
        w_steps = int((input_shape[2] - self.kernel_dim[1] + 1)/self.strides[1])

        # Each block or capsules has volume k_h * k_w * num_input_channels
        # There is one block for each output capsule
        blocks = []
        for h_step in range(h_steps): # iterate vertically
            row = h_step*self.strides[0]
            row_of_blocks = []
            for w_step in range(w_steps): # iterate horizontally
                col = w_step*self.strides[1]
                # pose_block shape: [batch_size, h_kernel, w_kernel, num_input_channels] + input_caps_dim
                pose_block = input_poses[:, row:row+self.kernel_dim[0], col:col+self.kernel_dim[1], :, :, :] 

                # Add two empty dims for new_im_height and new_im_width
                pose_block = tf.reshape(pose_block, [input_shape[0], 1, 1, self.kernel_dim[0], self.kernel_dim[1], input_shape[-2], input_shape[-1]])
                row_of_blocks.append(pose_block) # Create a row of pose_blocks

            # Append rows together to create a 2d matrix/image of pose blocks
            blocks.append(tf.concat(row_of_blocks, axis=2)) 

        blocks = tf.concat(blocks, axis=1) # Concat blocks from list into a tensor

        return blocks # shape: [batch_size, im_height=h_steps, im_width=w_steps, k_h, k_w, num_input_channels, input_caps_dim]

    def _get_votes(pose_blocks):
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
        votes = tf.matmul(blocks, self.kernel)

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

        # input poses shape [batch_size, input_height, input_width, num_input_channels, input_caps_dim[0], input_caps_dim[1]]
        # Input activations shape [batch_size, input_height, input_width, num_input_channels]

        # Calculate Votes from Poses
        ##############################
        blocks = self._get_pose_blocks(input_poses)

        # Add empty dim to allow multiple output channels
        # [batch_size, out_h, out_w, k_h, k_w, in_chan, in_caps_dim] -> [batch_size, out_h, out_w, 1, k_h, k_w, in_chan, in_caps_dim]
        blocks = tf.expand_dims(blocks, axis=3)

        votes = tf.matmul(blocks, self.kernel) # [batch_size, im_h, im_w, out_chan, k_h, k_w, in_chan] + out_caps_dim

        # Get Image Shape Attributes
        ##############################
        shape = tf.shape(votes)
        im_height = shape[1] # Output height
        im_width = shape[2] # Output width
        num_votes_per_capsule = shape[4]*shape[5]*shape[6]

        # Routing
        ############################

        # Reshape votes, flattening k_h, k_w and in_chan into one dimension
        votes = tf.reshape(votes, [-1, im_height, im_width, self.num_channels, num_votes_per_capsule] + self.capsule_dim)
        if self.routing == 'dynamic':
            # capsule_poses shape: [batch_size, im_height, im_width, num_out_channels] + capsule_dim
            capsule_poses = routing.dynamic_routing(votes) 
        elif self.routing == 'EM':
            raise ValueError('EM Routing Not Yet Implemented. Please select dynamic instead')
        else:
            raise ValueError('Was expecting either EM or dynamic for routing argument')

        # Calculate new capsule activations
        if self.activation == 'activation':
            raise ValueError('Sigmoid activation not yet supported for ConvCaps2D')
        else:
            # Flatten capsule poses into vectors
            shape = capsule_poses.shape
            vectors = tf.reshape(capsule_poses, shape=[-1, shape[1], shape[2]*shape[3]])
            if self.activation == 'squash':
                # Use squash function on capsule poses
                vectors = tools.squash(vectors)

            # Calculate length of each capsule vector
            capsule_activations = tf.norm(vectors, axis=-1)

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