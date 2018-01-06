# Color Quantization with Generative Adversarial Network

Color quantization is an important image processing methodology which can be applied in image displaying, 
image compressing and color printing. Traditional quantization methods 
are essentially based on clustering-based approach such as LBG algorithm and 
partitioning-based approach like median-cut algorithm. 

In this project, our contribution is we suggest a new Neural Network-based approach for image color quantization tasks.
Conditional Generative Adversarial Networks are built to achieve the palette generation and
the color reduction work. The trained Generative Adversarial Network will be capable of
generating customized palette from a predefined standard color palette. Normally, the
output of a well-trained generator would be three-channel color images. However, what we
want the generator to produce is a color palette rather than a color image. Inspired by
Zhang, we regard the palette as a convolution filter in the generative network and then
come up with a special structure for the generator. In our design, the palette can be updated
through back-propagation and extracted from the generator after the training

Compared to the traditional color quantization methods, we demonstrate that our network generates palette
which can make the reconstructed images have similar perceptual quality with traditional
color quantization algorithms.