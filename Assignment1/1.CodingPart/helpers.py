# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from numpy.fft import fft2, ifft2
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

def my_imfilter(image, filter):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter to an image. Return the filtered image.
    Inputs:
    - image -> numpy nd-array of dim (m, n, c)
    - filter -> numpy nd-array of odd dim (k, l)
    Returns
    - filtered_image -> numpy nd-array of dim (m, n, c)
    Errors if:
    - filter has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.asarray([0])
    
    if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
        raise ValueError('Only odd filters are allowed')
        
    #pad image edges with zeros to avoid dimension reduction
    pad_v = int((filter.shape[0] - 1) / 2)
    pad_h = int((filter.shape[1] - 1) / 2)
    
    if len(image.shape) == 2:
        image_padded = np.pad(image, [(pad_v, pad_v),(pad_h, pad_h)], mode = 'reflect')    
        filtered_image = np.zeros((image.shape[0], image.shape[1]))

        for x in range (image_padded.shape[0]):
            if x + filter.shape[0] > image_padded.shape[0]:
                break
            for y in range(image_padded.shape[1]):
                if y + filter.shape[1] > image_padded.shape[1]:
                    break
                filtered_image[x, y] = np.tensordot(image_padded[x:x + filter.shape[0],y:y+filter.shape[1]], filter)
    
    elif len(image.shape) == 3:
        image_padded = np.pad(image, [(pad_v, pad_v),(pad_h, pad_h),(0, 0)], mode = 'reflect')    
        filtered_image = np.zeros((image.shape[0], image.shape[1], 3))

        for x in range (image_padded.shape[0]):
            if x + filter.shape[0] > image_padded.shape[0]:
                break
            for y in range(image_padded.shape[1]):
                if y + filter.shape[1] > image_padded.shape[1]:
                    break
                for z in range(image_padded.shape[2]):
                    filtered_image[x, y, z] = np.tensordot(image_padded[x:x + filter.shape[0],y:y+filter.shape[1],z], filter)
        
        
    else: raise ValueError('Wrong image format')

    return filtered_image

def my_filter_fft(image, filter):
    if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
        raise ValueError('Only odd filters are allowed')
    
    if len(image.shape) == 2:
        image_fft = fft2(image)
        filter_fft = fft2(filter, s = image.shape)
        filtered_image = np.real(ifft2(image_fft * filter_fft))
        
    elif len(image.shape) == 3:
        image_fft_r = fft2(image[:,:,0])
        image_fft_g = fft2(image[:,:,1])
        image_fft_b = fft2(image[:,:,2])
        filter_fft = fft2(filter, s = image_fft_r.shape)
        filtered_image = np.real(np.stack((ifft2(image_fft_r * filter_fft),ifft2(image_fft_g * filter_fft),ifft2(image_fft_b * filter_fft)),axis = 2))
        
    else: raise ValueError('Wrong image format')

    return filtered_image

def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
    Inputs:
    - image1 -> The image from which to take the low frequencies.
    - image2 -> The image from which to take the high frequencies.
    - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

    Task:
    - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
    - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here:
    low_frequencies = my_imfilter(image1, kernel) # Replace with your implementation

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = image2 - my_imfilter(image2, kernel) # Replace with your implementation

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = low_frequencies + high_frequencies # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.
    hybrid_image = np.clip(hybrid_image, 0, 1)
    return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel = True)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
