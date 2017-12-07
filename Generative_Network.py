# coding: utf-8
import numpy as np
import scipy.misc
import tensorflow as tf


def net(image, rgb_palette):
    ### 输入的图像不用normalization [batch, height = 256, width = 256, channel = 3]
    with tf.variable_scope('generator'):
        with tf.name_scope('GN_layer1'):
            normalization = batch_norm(image, train=True)
            conv1_1_relu = conv_layer(normalization, 64, 3, 1, relu=True) #[batch, height = 256, width = 256, channel = 64]
            conv1_2_relu = conv_layer(conv1_1_relu, 64, 3, 1, relu=True)
            conv1_2norm = batch_norm(conv1_2_relu, train=True)

        with tf.name_scope('GN_layer2'):
            conv2_1_relu = conv_layer(conv1_2norm, 128, 3, 1, relu=True)
            conv2_2_relu = conv_layer(conv1_1_relu, 128, 3, 2, relu=True) 
            conv2_2norm = batch_norm(conv2_2_relu, train=True)

        with tf.name_scope('GN_layer3'):
            conv3_1_relu = conv_layer(conv2_2norm, 256, 3, 1, relu=True)
            conv3_2_relu = conv_layer(conv3_1_relu, 256, 3, 1, relu=True)
            conv3_3_relu = conv_layer(conv3_2_relu, 256, 3, 2, relu=True)
            conv3_3norm = batch_norm(conv3_3_relu, train=True)
        '''
        conv4_1 (Stride:1,pad:1 dilation: 1)> relu4_1 > conv4_2(same) > relu4_2 > conv4_3(same) > relu4_3 > conv4_3_norm
        tf.nn.atrous_conv2d(net, weights_init, rate, 'SAME')
        conv_layer_dila(net, num_filters, filter_size, rate, relu=True)
        '''
        with tf.name_scope('GN_layer4'):
            conv4_1_relu = conv_layer(conv3_3norm, 512, 3, 2, relu=True)
            conv4_2_relu = conv_layer_dila(conv4_1_relu, 512, 3, 1, relu=True)
            conv4_3_relu = conv_layer_dila(conv4_2_relu, 512, 3, 1, relu=True)
            conv4_3norm = batch_norm(conv4_3_relu, train=True)

        with tf.name_scope('GN_layer5'):
            conv5_1_relu = conv_layer(conv4_3norm, 512, 3, 1, relu=True)
            conv5_2_relu = conv_layer_dila(conv5_1_relu, 512, 3, 1, relu=True)
            conv5_3_relu = conv_layer_dila(conv5_2_relu, 512, 3, 1, relu=True)
            conv5_3norm = batch_norm(conv5_3_relu, train=True)

        with tf.name_scope('GN_layer6'):    
            conv6_1_relu = conv_tranpose_layer(conv5_3norm, 256, 3, 2)
            conv6_2_relu = conv_layer_dila(conv6_1_relu, 256, 3, 1, relu=True)
            conv6_3_relu = conv_layer_dila(conv6_2_relu, 256, 3, 1, relu=True)
            conv6_3norm = batch_norm(conv6_3_relu, train=True)   

        with tf.name_scope('GN_layer7'):    
            conv7_1_relu = conv_tranpose_layer(conv6_3norm, 256, 3, 2)
            conv7_2_relu = conv_layer(conv7_1_relu, 256, 3, 1, relu=True)
            conv7_3_relu = conv_layer(conv7_2_relu, 256, 3, 1, relu=True)
            conv7_3norm = batch_norm(conv7_3_relu, train=True)   

        with tf.name_scope('GN_layer8'):    
            conv8_1_relu = conv_tranpose_layer(conv7_3norm, 256, 3, 2)
            conv8_2_relu = conv_layer(conv8_1_relu, 256, 3, 1, relu=True)
            conv8_3_relu = conv_layer(conv8_2_relu, 256, 3, 1, relu=True) ### Output batch x 256 x 256 x 256


        with tf.name_scope('GN_Prob'):
            prob_distribution = tf.nn.softmax(conv8_3_relu)
            
            
        with tf.name_scope('mapping'):
            mapping_filter = tf.Variable(np.transpose(rgb_palette, [1,0]), dtype=tf.float32)
            mapping_filter = tf.reshape(mapping_filter, [1, 1, 256, 3])
            cq_image = tf.nn.conv2d(prob_distribution, mapping_filter, strides=[1,1,1,1], padding='SAME')

            cq_image  = tf.clip_by_value(cq_image , 0., 255., name=None)

        return cq_image
      

def conv_init_vars(net, out_channels, filter_size, transpose=False):
    '''
    
    According to the previous output, intialize the weight matrix.
    
    
    '''
    _, rows, cols, in_channels = [i.value for i in net.get_shape()] ### Obtain in_channels
    
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    
    # weights shape = [Kernal size, kernal size, output kernal, input kernal]
    
    xaiver_init_2d = tf.contrib.layers.xavier_initializer_conv2d()

    weights_init = tf.Variable(xaiver_init_2d(weights_shape), dtype=tf.float32)
    
    return weights_init


def bias_init_vars(outputfilter):

    bias_shape = [outputfilter]
    
    bias_init = tf.Variable(tf.constant(0.01, shape = bias_shape, dtype=tf.float32))
    
    return bias_init
    
    
def batch_norm(net, train=True):
    '''
    
    Apply Batch Normalization Function
    
    BN: Forward norm and then inverse norm.
    
    '''
    batch, rows, cols, channels = [i.value for i in net.get_shape()]### Shape Meaning: [batchsize, height, width, kernels]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True) ### Calculate the mean and variance of x.Output: One-dimension
    shift = tf.Variable(tf.zeros(var_shape)) ### Inverse Norm
    scale = tf.Variable(tf.ones(var_shape)) ### Inverse Norm
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift ### Applied Batch Normalization


def conv_layer(net, num_filters, filter_size, strides, relu=True):
    '''
    
    Apply convolution operation
    
    '''
    weights_init = conv_init_vars(net, num_filters, filter_size)
    bias_init = bias_init_vars(num_filters)
    strides_shape = [1, strides, strides, 1]                
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = tf.nn.bias_add(net, bias_init)
    
    if relu:
        net = lrelu(net)   
    
    return net        

def conv_layer_dila(net, num_filters, filter_size, rate, relu=True):
    '''
    
    Apply dilation convolution operation
    
    '''
    weights_init = conv_init_vars(net, num_filters, filter_size)
    #strides_shape = [1, strides, strides, 1]
    bias_init = bias_init_vars(num_filters)
    net = tf.nn.atrous_conv2d(net, weights_init, rate, 'SAME') # Dialation Convolution
    net = tf.nn.bias_add(net, bias_init)    
    if relu:
        net = lrelu(net)   
    
    return net   

def conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = conv_init_vars(net, num_filters, filter_size, transpose=True)
    bias_init = bias_init_vars(num_filters)
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])
    new_shape = [batch_size, new_rows, new_cols, num_filters]  
    tf_shape = tf.stack(new_shape)   
    strides_shape = [1,strides,strides,1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = tf.nn.bias_add(net, bias_init)
    return lrelu(net)

def lrelu(net, alpha = 0.2):
    return tf.nn.relu(net) - alpha * tf.nn.relu(-net)
            
if __name__ == '__main__':
    main()
