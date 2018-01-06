# coding:utf-8

from sklearn.cluster import KMeans
import skimage
import scipy.misc
import numpy as np
import skimage.color
import matplotlib.pyplot as plt

imagepath = 'C:\\Users\\Orion_Peng\\Pictures\\Saved Pictures\\dog.jpg'


def get_img(img_path):
    img = scipy.misc.imread(img_path, mode = 'RGB')
    return img

def rgb2lab(image):
    '''
    L range: 0 ~ 100
    a range: -128 ~ 127
    b range: -128 ~ 127
    
    '''
    lab_color = skimage.color.rgb2lab(image)
    return lab_color


def lab2rgb(image):
    rgb_color = skimage.color.lab2rgb(image)
    return rgb_color


def nearest_search(image, palette):
    '''
    Palette shape: (16, 16, 3)
    
    '''
    img_shape = image.shape
    height = img_shape[0] 
    width = img_shape[1] 
    new_img = np.zeros(img_shape)
    for i in range(height):
        for j in range(width):
            index = find_min_idx(np.sum((image[i, j, :] - palette)**2,2))
            new_img[i, j, :] = palette[index[0], index[1], :]
    
    return new_img


def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return np.int(k/ncol), k%ncol

def main():
    img = get_img(imagepath)
    img = scipy.misc.imresize(img, (256, 256))
    img_lab = rgb2lab(img)
    img_reshape = np.reshape(img_lab,(256 * 256, 3))

    kmeans = KMeans(n_clusters=256, random_state=0).fit(img_reshape)
    lab_palette = kmeans.cluster_centers_
    lab_palette = np.reshape(lab_palette,(16,16,3))
    rgb_palette = lab2rgb(lab_palette)

    img_rec = nearest_search(img_lab, lab_palette)
    img_rec = lab2rgb(img_rec)
    
    plt.figure(1)
    plt.imshow(img)
    plt.figure(2)
    plt.imshow(rgb_palette)
    plt.figure(3)
    plt.imshow(img_rec)
    plt.show()

if __name__ == '__main__':
    main()
