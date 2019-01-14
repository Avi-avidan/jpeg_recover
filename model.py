import tensorflow as tf
from base_network import ResNet_Model

class DfnNet:
    def __init__(self, train_y, train_x, bn_moment=0.99):
        self.bn_moment = bn_moment
        self.score = None                   # nn output, before argmax
        self.input_shape = [None,None,3]
        self.output_shape = [None,None,2]
        self.out_filters = self.output_shape[-1]
        
        self.tf_global_step = None
        self.tf_learning_rate = None
        with tf.name_scope("model_input"):
            self.x = train_x
        with tf.name_scope("model_output"):
            self.label = train_y
        
        self.train_step = None              # operation to run graph optimization
        self.loss = None                    # training loss - a scalar

        # masks prediction
        self.bmp_predict = None

        self.data_format = 'channels_last'
        self.build_nn()

    def build_nn(self):

        with tf.name_scope('parameters'):
            self.tf_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
            self.tf_learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            
            x_input = self.x
            # transpose to NCHW if needed
            if self.data_format == 'channels_first':
                x_input = tf.transpose(x_input, [0, 3, 1, 2])

        base_network = ResNet_Model(
            bottleneck=True,    # true for resnet 50
            num_filters=64,
            kernel_size=5,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=[3, 4, 6, 3],
            block_strides=[1, 2, 2, 2],
            resnet_version=1,
            data_format=self.data_format)

        resnet_outputs = base_network(x_input, training=True)

        res_1_out = resnet_outputs[0]
        res_2_out = resnet_outputs[1]
        res_3_out = resnet_outputs[2]
        res_4_out = resnet_outputs[3]
        average_pool = resnet_outputs[4]

        # squeeze number of channels, 2048->512
        average_pool = tf.layers.conv2d(
            inputs=average_pool,
            filters=512,
            kernel_size=1,
            padding='same',
            data_format=self.data_format,
            name='average_pool_channels'
        )

        with tf.name_scope('deconv_4'):
            if self.data_format == 'channels_last':
                multiples=[1, tf.shape(res_4_out)[1], tf.shape(res_4_out)[2], 1]
            else:
                multiples=[1, 1, tf.shape(res_4_out)[2], tf.shape(res_4_out)[3]]
                
            average_unpool = tf.tile(average_pool, multiples=multiples, name='average_unpool')
            rrb_4_1 = self.rrb_layer(res_4_out, 'rrb_4_1')
            cab_4 = self.cab_layer(rrb_4_1, average_unpool, 'cab_4')
            rrb_4_2 = self.rrb_layer(cab_4, 'rrb_4_2')
        
        rrb_3_2 = self.deconv_rrb(rrb_4_2, res_3_out, 3)
        rrb_2_2 = self.deconv_rrb(rrb_3_2, res_2_out, 2)
        rrb_1_2 = self.deconv_rrb(rrb_2_2, res_1_out, 1)
        rrb_1_3 = self.conv_layer(rrb_1_2, 3, 256, name='rrb_1_3')
        
        with tf.name_scope('upsample_output'):
            upsample_1 = self.upsample_output(rrb_1_3, 128, 'upsample_1')
            upsample_2 = self.upsample_output(upsample_1, 64, 'upsample_2')
        
        with tf.name_scope('nn_predict'):    
            self.bmp_m2 = self.concat_input(upsample_2, x_input, 2)
            self.bmp_m1 = self.concat_input(self.bmp_m2, x_input, 1)
            self.bmp_predict = self.concat_input(self.bmp_m1, x_input, 0)
        
        
        with tf.name_scope('nn_loss'):

            self.loss_1 = tf.reduce_mean(tf.math.squared_difference(self.bmp_m2,self.label), name='rmse_loss_2')
            self.loss_2 = tf.reduce_mean(tf.math.squared_difference(self.bmp_m1,self.label), name='rmse_loss_1')
            self.loss_3 = tf.reduce_mean(tf.math.squared_difference(self.bmp_predict,self.label), name='rmse_loss_0')

            self.loss = 1*self.loss_1 + 1*self.loss_2 + 5*self.loss_3

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_step = \
                    tf.train.AdamOptimizer(self.tf_learning_rate).minimize(self.loss, global_step=self.tf_global_step)


    def conv_layer(self, inputs, kernel, filters, name, use_bn=True, stride=1):
        # apply convolution
        x = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel,
            strides=stride,
            padding="same",
            data_format=self.data_format,
            name=(name+'_conv')
        )
        if use_bn:
            # apply batch norm
            x = tf.layers.batch_normalization(
                inputs=x,
                momentum=self.bn_moment,
                training=True,
                fused=True,
                axis=1 if self.data_format == 'channels_first' else 3,
                name=(name+'_batch_norm')
            )
        # apply relu
        x = tf.nn.relu(
            x,
            name=(name+'_relu')
        )
        return x

    @staticmethod
    def pool_layer(x):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def cab_layer(self, x, upsampled, name):
        y = tf.concat([x, upsampled], axis=1 if self.data_format == 'channels_first' else 3)
        y = tf.reduce_mean(y, axis=[2, 3] if self.data_format == 'channels_first' else [1, 2], keepdims=True)
        y = self.conv_layer(y, 1, 512, name+'_conv_1')
        y = tf.layers.conv2d(
            inputs=y,
            filters=512,
            kernel_size=1,
            padding='same',
            data_format=self.data_format,
            name=name+'conv_2'
        )
        y = tf.nn.sigmoid(y, name=name+'sigmoid')
        x = tf.multiply(x, y)
        return tf.add(x, upsampled, name=name+'_add')

    def rrb_layer(self, x, name):
        decoded = tf.layers.conv2d(
            inputs=x,
            filters=512,
            kernel_size=(1, 1),
            padding="same",
            data_format=self.data_format,
            name=(name+'_conv')
        )
        refined = self.conv_layer(decoded, 3, 512, name+'refine')
        refined = tf.layers.conv2d(
            inputs=refined,
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            data_format=self.data_format,
            name=(name+'_decode')
        )
        summed_up = decoded + refined

        return tf.nn.relu(summed_up, name='rrb_relu')

    def upsample_output(self, lay_in, filters, name):
        return tf.layers.conv2d_transpose(
            inputs=lay_in,
            filters=filters,
            strides=(2, 2),
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            data_format=self.data_format,
            padding='same',
            name=name
        )
    
    def deconv_rrb(self, rrb_deconv, res_in, num):
        with tf.name_scope('deconv_' + str(num)):
            unpool_x = tf.layers.conv2d_transpose(
                inputs=rrb_deconv,
                filters=512,
                strides=(2, 2),
                kernel_size=(3, 3),
                activation=tf.nn.relu,
                data_format=self.data_format,
                padding='same',
                name='unpool_' + str(num)
            )
            rrb_x_1 = self.rrb_layer(res_in, 'rrb_' + str(num) + '_1')
            cab_x = self.cab_layer(rrb_x_1, unpool_x, 'cab_' + str(num))
            return self.rrb_layer(cab_x, 'rrb_' + str(num) + '_2')

    
    def concat_input(self, upsample_2, x_input, num):
        with tf.name_scope('conc_' + str(num)):
            concatenated = tf.concat(
                [upsample_2, x_input],
                axis=1 if self.data_format == 'channels_first' else 3,
                name='conc_input_' + str(num)
            )
            conv_6_1 = self.conv_layer(concatenated, 3, 32, name='conc_conv_' + str(num) + '_1')
            conv_6_2 = self.conv_layer(conv_6_1, 3, 32, name='conc_conv_' + str(num) + '_2')
            conv_6_3 = self.conv_layer(conv_6_2, 3, 32, name='conc_conv_' + str(num) + '_3')

            score = tf.layers.conv2d(
                inputs=conv_6_3,
                filters=self.out_filters,
                kernel_size=1,
                padding="same",
                data_format=self.data_format,
                name='score_conv_' + str(num)
            )
        
            if self.data_format == 'channels_first':
                score = tf.transpose(self.score, [0, 2, 3, 1])
            return score
           
    @staticmethod
    def deconv_layer(x, w_shape, name):
        return tf.layers.conv2d_transpose(
            inputs=x,
            filters=w_shape[2],
            kernel_size=w_shape[0:2],
            padding='same',
            name=(name+'conv_transpose')
        )
