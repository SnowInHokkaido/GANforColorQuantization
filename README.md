# Color Quantization with Generative Adversarial Network

Normally, there are 2 steps in color quantization: first, define a palette based on 
the colors of the image and then for each pixel, replace its color with a closest color 
in the palette to generate a new image. In this project, a conditional GAN would be 
trained. The generative network(GN) will have a structure similar to the one in Pix2pix. 
The input of the generative network will be a 24 bit color image and the output will be 
a palette (i.e. 256-color palette). The colors of the original image will be replaced by 
the ones in the generated palette by means of nearest neighbor search. After that, the 
modified image will be inputted into the discriminative network (DN) and DN will compare 
the ground truth and the modified image, to estimate which one is the ground truth. DN 
will improve decision boundary according to cross entropy loss function. GN will improve 
weighting through backpropagation and get new color palette in next round. The whole 
process would iterate a thousand times. The training set consist of 500 24-bit color 
images. 
