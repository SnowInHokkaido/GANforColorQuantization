# coding: utf-8
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
from argparse import ArgumentParser
import Generative_Network
import Discriminative_Network
import os

ORIGINALPALETTE = 'uv_palette.npy'
TRAININGDATAPATH = 'train_data.npy'
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
BATCHSIZE = 8
ITERATIONS = 100000
TRAININGRATIO = 3
LEARNINGRATE = 1e-3

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--palettepath',
            dest='palettepath', help='load the original color palette',
            metavar='ORIGINALPALETTE', default=ORIGINALPALETTE)
    parser.add_argument('--trainingdatapath',
            dest='trainingdatapath', help='training data in npy format',
            metavar='TRAININGDATAPATH', default=TRAININGDATAPATH)
    parser.add_argument('--vggpath',
            dest='vggpath', help='load pre-trained VGG19 model',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--batchsize', type=int,
            dest='batchsize', help='define the size of batch in training',
            metavar='BATCHSIZE', default=BATCHSIZE)
    parser.add_argument('--trainingratio', type=int,
            dest='trainingratio', help='define the iteration ratio between DN and GN',
            metavar='TRAININGRATIO', default=TRAININGRATIO)
    parser.add_argument('--learningrate', type=int,
            dest='learningrate', help='define the learning rate of Adamoptimizaer',
            metavar='LEARNINGRATE', default=LEARNINGRATE) 
    
    return parser

def g_network(image, rgb_palette, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    fake_image = Generative_Network.net(image, rgb_palette)
    return fake_image

def d_network(image, weights, reuse = False):
    if reuse:
        tf.get_variable_scope().reuse_variables()    
    prob, logits = Discriminative_Network.discriminator(image, weights)   
    return prob, logits

def next_batch(num, data):
    '''
    Return a total of `num` random samples
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle_color = np.array([rgb2yuv(data[i]) for i in idx]) ### RGB Space
    data_shuffle_gray = np.array([np.reshape(rgb2yuv(data[i])[:,:,0], [128, 128, 1]) for i in idx]) ### Gray Space
    return data_shuffle_color, data_shuffle_gray

def rgb2yuv(image):
    yuv_color = skimage.color.rgb2yuv(image)
    return yuv_color

def yuv2rgb(image):
    rgb_color = skimage.color.yuv2rgb(image)
    return rgb_color

def main():
    parser = build_parser()
    options = parser.parse_args()
    
    #Palette loading
    palettepath = options.palettepath
    orig_palette = np.load(palettepath)
    
    #Training Data Preparation
    training_path = options.trainingdatapath
    training_data = np.load(training_path)
    imagelist = []
    for i in range(295):
        imagelist.append(np.squeeze(training_data[i,:,:,:]))
    
    #VGG Model Loading
    net_path = options.vggpath
    weights, mean_pixel = Discriminative_Network.load_net(net_path)
    
    learningrate = options.learningrate
    batch_size = options.batchsize
    iterations = options.iterations
    trainingratio = options.trainingratio
    
    print('Data loading completed')

    g = tf.Graph()
    with g.as_default():

        X = tf.placeholder(tf.float32, [batch_size, 128, 128, 1])
        Y = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])

        with tf.name_scope('generating'):
            G_sample = g_network(X, orig_palette)

        with tf.name_scope('discriminating'):
            mean_tensor = tf.cast(np.reshape(mean_pixel, [1,1,1,3]), tf.float32)
            #Y_ = Y - mean_tensor
            #G_sample_ = G_sample - mean_tensor       
            D_real, D_logit_real = d_network(Y, weights) 
            D_fake, D_logit_fake = d_network(G_sample, weights)

        with tf.name_scope('D_loss'):
            D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
            D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
            D_loss = D_loss_real + D_loss_fake

        with tf.name_scope('G_loss'):
            G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        tvar = tf.trainable_variables()
        dvar = [var for var in tvar if 'discriminator' in var.name] # Find variable in DN
        gvar = [var for var in tvar if 'generator' in var.name] # Find variable in GB

        with tf.name_scope('train'):
            d_train = tf.train.AdamOptimizer(learningrate).minimize(D_loss, var_list=dvar)
            g_train = tf.train.AdamOptimizer(learningrate).minimize(G_loss, var_list=gvar)

        print('Graph establised')

        
    with tf.Session(graph = g) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        for i in range(iterations): # Train ratio: DN/GN = 100/1
            print('Training Step:' + str(i+1))
            
            batch_img_color, batch_img_gray = next_batch(batch_size, imagelist)

            if i % trainingratio == 0:
                _, G_loss_curr, samples = sess.run([g_train, G_loss, G_sample], feed_dict={X:batch_img_gray, Y: batch_img_color})
                if not os.path.isdir('tmp_output'):
                    os.mkdir('tmp_output')                    
                for index in range(batch_size):
                    tmp_name = str(index)
                    tmp_img = yuv2rgb(np.squeeze(samples[index,:,:,:]))
                    scipy.misc.imsave('tmp_output/iteration_' + str(i) +'_' + tmp_name +'.jpg', tmp_img)
                
            _, D_loss_curr = sess.run([d_train, D_loss], feed_dict={X:batch_img_gray, Y: batch_img_color})
        

            if i % trainingratio == 0:
                print(D_loss_curr, G_loss_curr)      
              
        save_path = saver.save(sess, "/model.ckpt")

if __name__ == '__main__':
    main()
        
               
