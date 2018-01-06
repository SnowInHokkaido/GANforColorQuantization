import skimage
import scipy.misc
import numpy as np
import skimage.color
import matplotlib.pyplot as plt

%matplotlib inline

imagepath = 'C:\\Users\\Orion_Peng\\Pictures\\Saved Pictures\\dog.jpg'

def get_img(img_path):
    img = scipy.misc.imread(img_path, mode = 'RGB')
    return img

def mediancut(rgb,count,ret=[],flag=1):
    """
    median cut takes rgb values of an image as np array "rgb"
    count = 0 to generate 256 colors; 4 to get 16 colors;
    ret=[]
    flag =1 for the function to work
    returns the ret as a list of color pallets, in RGB
    """
    if flag==1:
        ret = []
    if count < 8:
        
        col=np.array([np.ptp(rgb[:,0]),np.ptp(rgb[:,1]),np.ptp(rgb[:,2])]).argsort()[2]
        rgb=rgb[rgb[:,col].argsort()]
        rgb1=rgb[:(int)(len(rgb)/2),:]
        rgb2=rgb[int(len(rgb)/2):,:]
        (mediancut(rgb1,count+1,ret,0))
        (mediancut(rgb2,count+1,ret,0))
        return ret
    else:
        ret.append([(int)(np.mean(rgb[:,0])),(int)(np.mean(rgb[:,1])),(int)(np.mean(rgb[:,2]))])        

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
    rgb = np.reshape(img, [256*256, 3])
    mc = mediancut(rgb,0,[],1)

    rgb_palette = np.reshape(mc, [16, 16, 3]).astype(np.uint8)
    lab_palette = rgb2lab(rgb_palette)
    img_lab = rgb2lab(img)
    img_rec = nearest_search(img_lab,lab_palette)
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
    
