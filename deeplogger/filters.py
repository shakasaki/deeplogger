# a set of filters that can be applied to the images
import numpy as np
import cv2

def dnorm(x, mu, sd):
    '''Calculates the normal distribution of a given value.
    Args:
        x (float): The value to be evaluated.
        mu (float): The mean of the distribution.
        sd (float): The standard deviation of the distribution.
    Returns:
        float: The normal distribution of the value.'''
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    '''Generates a Gaussian kernel with a given size and standard deviation.
    Args:
        size (int): The size of the kernel.
        sigma (int): The standard deviation of the kernel.
    Returns:
        np.array: The Gaussian kernel.'''
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D


def convolution(image: np.array,
                kernel:np.array,
                average=False):
    '''Applies a convolution to the image using the kernel.
    Args:
        image (np.array): The image to be convolved.
        kernel (np.array): The kernel to be applied.
        average (bool): Whether to average the output.
    Returns:
        np.array: The convolved image.
        '''
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height,
    pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output


def gaussian_blur(image: np.array, 
                  kernel_size: int = 1):
    '''Applies a Gaussian kernel to the image through convolution.
    Args:
        image (np.array): The image to be blurred.
        kernel_size (int): The size of the kernel to be used.
    Returns:
        np.array: The blurred image.'''
    kernel = gaussian_kernel(kernel_size, sigma=np.sqrt(kernel_size))
    conv = convolution(image, kernel, average=False)
    # normalize the image while avoiding division by zero or invalid value divide
    # check if maximum is positive and normalize the image
    if np.max(conv) > 0:
        conv = conv / np.max(conv)
    # replace nans with 0
    return np.nan_to_num(conv)

# filter that takes an image with binary pixels and applies a neighbor algorithm to increase the pixel width
def neighbor_filter(image, kernel_size: int = 1):
    '''Applies a neighbor filter to the image. This is good to increase the pixel width.
    Args:
        image (np.array): The image to be filtered.
        kernel_size (int): The size of the kernel to be used.
    Returns:
        np.array: The filtered image.'''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image = image.astype(np.uint8)
    return cv2.dilate(image, kernel, iterations=1)




