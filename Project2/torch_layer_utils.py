#useful util functions implemented with pytorch

import torch
from torch import nn
import numpy as np
"""
Image gradients are needed for both SIFT and the Harris Corner Detector, so we
implement the necessary code only once, here.
"""


class ImageGradientsLayer(torch.nn.Module):
    """
    ImageGradientsLayer: Compute image gradients Ix & Iy. This can be
    approximated by convolving with Sobel filter.
    """
    def __init__(self):
        super(ImageGradientsLayer, self).__init__()

        # Create convolutional layer
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,
            bias=False, padding=(1,1), padding_mode='zeros')

        # Instead of learning weight parameters, here we set the filter to be
        # Sobel filter
        self.conv2d.weight = get_sobel_xy_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of ImageGradientsLayer. We'll test with a
        single-channel image, and 1 image at a time (batch size = 1).

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, (num_image, 2, height, width)
            tensor for Ix and Iy, respectively.
        """
        return self.conv2d(x)


def get_gaussian_kernel(ksize=7, sigma=5) -> torch.nn.Parameter:
    """
    Generate a Gaussian kernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: torch.nn.Parameter of size [ksize, ksize]
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    k = ksize
    mean = k // 2
    kern = np.zeros(k)

    # Populate one row of density values
    for x in range(k):
        p = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(1/(2*np.power(sigma,2))) * (np.power((x-mean),2)))
        kern[x] = p

    # Create k by k matrix
    kernel = np.outer(kern, kern)

    # Find total sum of kernel
    rowsum = 0
    for i in range(len(kern)):
        for j in range(len(kern)):
            rowsum += kernel[i][j]

    # Divide each item in the final kernel by rowsum
    for i in range(len(kern)):
        for j in range(len(kern)):
            kernel[i][j] /= rowsum

    kernel = torch.Tensor(kernel)
    kernel = nn.Parameter(kernel)

    # raise NotImplementedError('`get_gaussian_kernel` need to be '
    #     + 'implemented')

    ### END OF STUDENT CODE ####
    ############################
    return kernel


def get_sobel_xy_parameters() -> torch.nn.Parameter:
    """
    Populate the conv layer weights for the Sobel layer (image gradient
    approximation).

    There should be two sets of filters: each should have size (1 x 3 x 3)
    for 1 channel, 3 pixels in height, 3 pixels in width. When combined along
    the batch dimension, this conv layer should have size (2 x 1 x 3 x 3), with
    the Sobel_x filter first, and the Sobel_y filter second.

    Args:
    -   None
    Returns:
    -   kernel: Torch parameter representing (2, 1, 3, 3) conv filters
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    Sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # print("sobel x shape", Sobel_x.shape)
    # print("sobel x", Sobel_x)

    newx = np.reshape(Sobel_x, (1, 1, Sobel_x.shape[0], Sobel_x.shape[1]))
    newy = np.reshape(Sobel_y, (1, 1, Sobel_y.shape[0], Sobel_y.shape[1]))

    # print("newx shape", newx.shape)
    # print("newx", newx)

    kern = np.concatenate((newx, newy))

    # print("kern shape", kern.shape)
    # print("kern", kern)

    kernel = torch.nn.Parameter(torch.Tensor(kern))

    # raise NotImplementedError('`get_sobel_xy_parameters` need to be '
    #     + 'implemented')

    ### END OF STUDENT CODE ####
    ############################

    return kernel
