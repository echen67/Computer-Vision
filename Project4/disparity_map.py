"""
This file holds the main code for disparity map calculations
"""
import torch
import numpy as np

from typing import Callable, Tuple


def calculate_disparity_map(left_img: torch.Tensor,
                            right_img: torch.Tensor,
                            block_size: int,
                            sim_measure_function: Callable,
                            max_search_bound: int = 50) -> torch.Tensor:
    """
    Calculate the disparity value at each pixel by searching a small 
    patch around a pixel from the left image in the right image

    Note: 
    1.  It is important for this project to follow the convention of search
            input in left image and search target in right image
    2.  While searching for disparity value for a patch, it may happen that there
            are multiple disparity values with the minimum value of the similarity
            measure. In that case we need to pick the smallest disparity value.
            Please check the numpy's argmin and pytorch's argmin carefully.
            Example:
            -- diparity_val -- | -- similarity error --
            -- 0               | 5 
            -- 1               | 4
            -- 2               | 7
            -- 3               | 4
            -- 4               | 12

            In this case we need the output to be 1 and not 3.
    3. The max_search_bound is defined from the patch center.

    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                                C will be >= 1.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   block_size: the size of the block to be used for searching between
                                    left and right image
    -   sim_measure_function: a function to measure similarity measure between
                                                        two tensors of the same shape; returns the error value
    -   max_search_bound: the maximum horizontal distance (in terms of pixels) 
                                                to use for searching
    Returns:
    -   disparity_map: The map of disparity values at each pixel. 
                                         Tensor of shape (H-2*(block_size//2),W-2*(block_size//2))
    """

    assert left_img.shape == right_img.shape
    disparity_map = torch.zeros(1) #placeholder, this is not the actual size
    ############################################################################
    # Student code begin
    ############################################################################

    H = left_img.shape[0]
    W = left_img.shape[1]
    C = left_img.shape[2]
    disparity_map = torch.zeros((H-2*(block_size//2), W-2*(block_size//2)))
    
    for h in range(block_size//2, H-block_size//2):
        for w in range(block_size//2, W-block_size//2):
            search = min(max_search_bound, w-block_size//2+1)
            minSim = float('inf')
            dis = 0
            patch1 = left_img[h-block_size//2:h+block_size//2+1, w-block_size//2:w+block_size//2+1, :]
            patch2 = right_img[h-block_size//2:h+block_size//2+1, w-block_size//2:w+block_size//2+1, :]
            sim = sim_measure_function(patch1, patch2)
            if sim < minSim:
                minSim = sim
                dis = 0
            for d in range(search):
                patch1 = left_img[h-block_size//2:h+block_size//2+1, w-block_size//2:w+block_size//2+1, :]
                patch2 = right_img[h-block_size//2:h+block_size//2+1, w-block_size//2-d:w+block_size//2+1-d, :]
                sim = sim_measure_function(patch1, patch2)  # tensor of single value
                if sim < minSim:
                    minSim = sim
                    dis = d
            disparity_map[h-block_size//2][w-block_size//2] = dis

    # raise NotImplementedError('calculate_disparity_map not implemented')

    ############################################################################
    # Student code end
    ############################################################################
    return disparity_map

def calculate_cost_volume(left_img: torch.Tensor,
                                                    right_img: torch.Tensor,
                                                    max_disparity: int,
                                                    sim_measure_function: Callable,
                                                    block_size: int = 9):
    """
    Calculate the cost volume. Each pixel will have D=max_disparity cost values
    associated with it. Basically for each pixel, we compute the cost of
    different disparities and put them all into a tensor.

    Note: 
    1.  It is important for this project to follow the convention of search
            input in left image and search target in right image
    2.  If the shifted patch in the right image will go out of bounds, it is
            good to set the default cost for that pixel and disparity to be something
            high(we recommend 255), so that when we consider costs, valid disparities will have a lower
            cost. 

    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                                C will be 1 or 3.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   max_disparity:  represents the number of disparity values we will consider.
                                    0 to max_disparity-1
    -   sim_measure_function: a function to measure similarity measure between
                                    two tensors of the same shape; returns the error value
    -   block_size: the size of the block to be used for searching between
                                    left and right image
    Returns:
    -   cost_volume: The cost volume tensor of shape (H,W,D). H,W are image
                                dimensions, and D is max_disparity. cost_volume[x,y,d] 
                                represents the similarity or cost between a patch around left[x,y]  
                                and a patch shifted by disparity d in the right image. 
    """
    #placeholder
    H = left_img.shape[0]
    W = right_img.shape[1]
    cost_volume = torch.zeros(H, W, max_disparity)
    ############################################################################
    # Student code begin
    ############################################################################

    # Pad disparity map to make it same size as images and cost_volume
    disparity_map = calculate_disparity_map(left_img, right_img, block_size, sim_measure_function, max_disparity)
    temp = torch.ones(H, W)
    temp = temp.new_full((H, W), 255)
    temp[block_size//2:H-block_size//2, block_size//2:W-block_size//2] = disparity_map
    disparity_map = temp

    for h in range(H):
        for w in range(W):
            for d in range(max_disparity):
                # Consider what happens if patch goes out of bounds
                if w < block_size//2 or w > W-block_size//2-1 or h < block_size//2 or h > H-block_size//2-1 or d > w-block_size//2:
                    penalty = 255
                    sim = 0
                else:
                    # Conpare similarity between patch from left image and shifted patches from right image
                    patch1 = left_img[h-block_size//2:h+block_size//2+1, w-block_size//2:w+block_size//2+1, :]
                    patch2 = right_img[h-block_size//2:h+block_size//2+1, w-block_size//2-d:w+block_size//2+1-d, :]
                    # Handle even block_size
                    if block_size % 2 == 0:
                        patch1 = left_img[h-block_size//2+1:h+block_size//2+1, w-block_size//2+1:w+block_size//2+1, :]
                        patch2 = right_img[h-block_size//2+1:h+block_size//2+1, w-block_size//2+1-d:w+block_size//2+1-d, :]
                    sim = sim_measure_function(patch1, patch2)
                    
                    # Compare disparity of current pixel to disparity of neighbors
                    patch1 = torch.ones((block_size, block_size))
                    patch1 = patch1.new_full((block_size, block_size), disparity_map[h][w])
                    patch2 = disparity_map[h-block_size//2:h+block_size//2+1, w-block_size//2:w+block_size//2+1]
                    # Handle even block_size
                    if block_size % 2 == 0:
                        patch2 = disparity_map[h-block_size//2+1:h+block_size//2+1, w-block_size//2+1:w+block_size//2+1]
                    penalty = sim_measure_function(patch1, patch2)
                cost_volume[h][w][d] = sim + penalty

    ############################################################################
    # Student code end
    ############################################################################
    return cost_volume
