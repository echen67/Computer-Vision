import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
    '''
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    '''

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    scaler = StandardScaler()

    # Training Set
    newpath = dir_name + 'train/'
    folders = os.listdir(newpath)
    for f in folders:
        items = os.listdir(newpath + f)
        for i in items:
            img = Image.open(newpath + f + '/' + i)
            img = img.convert(mode="L")
            img = np.array(list(img.getdata()))
            img = img / 255.0
            img = np.reshape(img, (-1, 1))
            scaler.partial_fit(img)

    # Test Set
    newpath = dir_name + 'test/'
    folders = os.listdir(newpath)
    for f in folders:
        items = os.listdir(newpath + f)
        for i in items:
            img = Image.open(newpath + f + '/' + i)
            img = img.convert(mode="L")
            img = np.array(list(img.getdata()))
            img = img / 255.0
            img = np.reshape(img, (-1, 1))
            scaler.partial_fit(img)

    mean = scaler.mean_
    std = scaler.scale_

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
