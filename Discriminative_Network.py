import tensorflow as tf
import numpy as np
import scipy.io

#Need to download the vgg19 in the current dir
model_file_path = 'imagenet-vgg-verydeep-19.mat'

def conv_layer(input, weights, relu=True):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
	
	if relu:
        conv = tf.nn.relu(conv)
    return conv

def pool_layer(input):
	pool = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    return pool

def get_kernels(weights,i)
	kernels = weights[i][0][0][2][0][0]
	kernels = np.transpose(kernels, (1, 0, 2, 3))

def net(image):

	#Load vgg19
	data = scipy.io.loadmat(model_file_path)
	
	#need to recalculate training data mean_pixel
	#self.mean_pixel = np.array([123.68, 116.779, 103.939])
	
	#Get all the layers weights
	weights = data['layers'][0]
	

	'''
	[i] represent the index of weights
	[0]'conv1_1', [1]'relu1_1', [2]'conv1_2', [3]'relu1_2', [4]'pool1',
	
	'''
	#For each block, do conv, relu and pool
	kernels = get_kernels(weights,0)	
	conv1_1_relu = conv_layer(image, kernels, relu=True)
	kernels = get_kernels(weights,2)
	conv1_2_relu = conv_layer(conv1_1_relu, kernels, relu=True)
	pool1 = pool(conv1_2_relu)
	
	'''
	
	[5]'conv2_1', [6]'relu2_1', [7]'conv2_2', [8]'relu2_2', [9]'pool2',
	
	'''
	
	kernels = get_kernels(weights,5)
	conv2_1_relu = conv_layer(pool1, kernels, relu=True)
	kernels = get_kernels(weights,7)
	conv2_2_relu = conv_layer(conv2_1_relu, kernels, relu=True)
	pool2 = pool(conv2_2_relu)

	'''
	
	[10]'conv3_1', [11]'relu3_1', [12]'conv3_2', [13]'relu3_2', [14]'conv3_3', [15]'relu3_3', [16]'conv3_4', [17]'relu3_4', [18]'pool3',
	
	'''
	
	kernels = get_kernels(weights,10)
	conv3_1_relu = conv_layer(pool2, kernels, relu=True)
	kernels = get_kernels(weights,12)
	conv3_2_relu = conv_layer(conv3_1_relu, kernels, relu=True)
	kernels = get_kernels(weights,14)
	conv3_3_relu = conv_layer(conv3_2_relu, kernels, relu=True)
	kernels = get_kernels(weights,16)
	conv3_4_relu = conv_layer(conv3_3_relu, kernels, relu=True)
	pool3 = pool(conv3_4_relu)
	
	'''
	
	[19]'conv4_1', [20]'relu4_1', [21]'conv4_2', [22]'relu4_2', [23]'conv4_3', [24]'relu4_3', [25]'conv4_4', [26]'relu4_4', [27]'pool4',

	'''
	
	kernels = get_kernels(weights,19)
	conv4_1_relu = conv_layer(pool3, kernels, relu=True)
	kernels = get_kernels(weights,21)
	conv4_2_relu = conv_layer(conv4_1_relu, kernels, relu=True)
	kernels = get_kernels(weights,23)
	conv4_3_relu = conv_layer(conv4_2_relu, kernels, relu=True)
	kernels = get_kernels(weights,25)
	conv4_4_relu = conv_layer(conv4_3_relu, kernels, relu=True)
	pool4 = pool(conv4_4_relu)
	
	'''
	
	[28]'conv5_1', [29]'relu5_1', [30]'conv5_2', [31]'relu5_2', [32]'conv5_3', [33]'relu5_3', [34]'conv5_4', [35]'relu5_4'
	
	'''
	
	kernels = get_kernels(weights,28)
	conv5_1_relu = conv_layer(pool4, kernels, relu=True)
	kernels = get_kernels(weights,30)
	conv5_2_relu = conv_layer(conv5_1_relu, kernels, relu=True)
	kernels = get_kernels(weights,32)
	conv5_3_relu = conv_layer(conv5_2_relu, kernels, relu=True)
	kernels = get_kernels(weights,34)
	conv5_4_relu = conv_layer(conv5_3_relu, kernels, relu=True)
	
	#only return the last layers results and further use for pix2pix.py 
	return conv5_4_relu
		
	
	
	
	