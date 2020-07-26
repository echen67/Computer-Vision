'''
Contains functions with different data transforms
'''

import numpy as np
import torchvision.transforms as transforms

from typing import Tuple


def get_fundamental_transforms(inp_size: Tuple[int, int],
                               pixel_mean: np.array,
                               pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the core transforms needed to feed the images to our model

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean  of the raw dataset
  - pixel_std: the standard deviation of the raw dataset
  Returns:
  - fundamental_transforms: transforms.Compose with the fundamental transforms
  '''

  fundamental_transforms = None

  #############################################################################
  # Student code begin
  #############################################################################

  fundamental_transforms = transforms.Compose([
      transforms.Resize(inp_size),
      transforms.ToTensor(),
      transforms.Normalize(pixel_mean, pixel_std),
    ])

  #############################################################################
  # Student code end
  #############################################################################
  return fundamental_transforms


def get_data_augmentation_transforms(inp_size: Tuple[int, int],
                                     pixel_mean: np.array,
                                     pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the data augmentation + core transforms needed to be applied on the
  train set

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean  of the raw dataset
  - pixel_std: the standard deviation of the raw dataset
  Returns:
  - aug_transforms: transforms.Compose with all the transforms
  '''

  aug_transforms = None

  #############################################################################
  # Student code begin
  #############################################################################

  aug_transforms = transforms.Compose([
      # Add data augmentation transforms here
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(hue=0.25),      
      # Copy over fundamental transforms here
      transforms.Resize(inp_size),
      transforms.ToTensor(),
      transforms.Normalize(pixel_mean, pixel_std),
    ])

  #############################################################################
  # Student code end
  #############################################################################
  return aug_transforms
