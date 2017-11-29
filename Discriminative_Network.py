import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc

VGG19_LAYERS=(
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    weights = data['layers'][0]
    #use the original code in vgg19.py
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    return weights, mean_pixel 

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel

def unprocess(image, mean_pixel):
    return image + mean_pixel

def net_preloaded(input_image, weights):
    net = {}
    current = input_image

    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            #use the original code in vgg19.py
            kernels, bias = weights[i][0][0][0][0]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)

            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net

def batch_norm(input):
    epsilon = 1e-3
    batch_mean,batch_var=tf.nn.moments(input,[0])
    normalized=tf.nn.batch_normalization(input,mean=batch_mean,variance=batch_var,offset=None,scale=None,variance_epsilon=epsilon)
    return normalized

def discrimator(input_image,netpath):
    weights, mean_pixel = load_net(netpath)
    img = preprocess(input_image, mean_pixel)
    img = np.array([img]).astype(np.float32)
    
    with tf.variable_scope("VGG19_LAYERS"):
        relu5_4 = net_preloaded(img, weights)['relu5_4']

    with tf.variable_scope('FC_Softmax'):
    #block6 : [batch, 16, 16, 512] => [batch, 4096]
    #if i check the shape of relu5_4, the result must be [batch, 16, 16, 512] not [batch, 8, 8, 512]
        with tf.variable_scope('FC6') as scope:
            shape=int(np.prod(relu5_4.get_shape()[1:]))
            weight=tf.Variable(tf.truncated_normal([shape,4096],dtype=tf.float32,stddev=1e-1))
            bias=tf.Variable(tf.constant(1.0,shape=[4096],dtype=tf.float32))
            flat=tf.reshape(relu5_4,[-1,shape])
            fc6=tf.nn.bias_add(tf.matmul(flat,weight),bias)
            fc6=tf.nn.relu(fc6) 
            fc6=batch_norm(fc6)
        #block7 : [batch, 4096] => [batch, 1024]
        with tf.variable_scope('FC7') as scope:	
            weight=tf.Variable(tf.truncated_normal([4096,4096],dtype=tf.float32,stddev=1e-1))
            bias=tf.Variable(tf.constant(1.0,shape=[4096],dtype=tf.float32))	   
            fc7=tf.nn.bias_add(tf.matmul(fc6,weight),bias)
            fc7=tf.nn.relu(fc7) 
            fc7=batch_norm(fc7)
        #block8 : [batch, 1024] => [batch, 256]
        with tf.variable_scope('FC8') as scope:	
            weight=tf.Variable(tf.truncated_normal([4096,256],dtype=tf.float32,stddev=1e-1))
            bias=tf.Variable(tf.constant(1.0,shape=[256],dtype=tf.float32))	   
            fc8=tf.nn.bias_add(tf.matmul(fc7,weight),bias)
            fc8=tf.nn.relu(fc8) 
            fc8=batch_norm(fc8)
        #block8 : [batch, 256] => [batch, 2]
        with tf.variable_scope('Softmax9') as scope:
            weight=tf.Variable(tf.truncated_normal([256,2],dtype=tf.float32,stddev=1e-1))
            bias=tf.Variable(tf.constant(1.0,shape=[2],dtype=tf.float32))
            softmax = tf.nn.bias_add(tf.matmul(fc8,weight),bias)   
            softmax = tf.nn.softmax(softmax)  

    return softmax


def main():
    #net and image both in current dir
    net_path = 'imagenet-vgg-verydeep-19.mat'
    img_path = 'example.jpg'

    img = scipy.misc.imread(img_path)
    resize = scipy.misc.imresize(img, (256, 256))

    softmax = discrimator(resize, net_path)
    
    with tf.Session() as sess:       
        sess.run(tf.global_variables_initializer())
        output = softmax.eval()

if __name__ == '__main__':
    main()