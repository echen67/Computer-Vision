#!/usr/bin/python3

import numpy as np

def create_Gaussian_kernel(cutoff_frequency):
  """
  Returns a 2D Gaussian kernel using the specified filter size standard
  deviation and cutoff frequency.

  The kernel should have:
  - shape (k, k) where k = cutoff_frequency * 4 + 1
  - mean = floor(k / 2)
  - standard deviation = cutoff_frequency
  - values that sum to 1

  Args:
  - cutoff_frequency: an int controlling how much low frequency to leave in
	the image.
  Returns:
  - kernel: numpy nd-array of shape (k, k)

  HINT:
  - The 2D Gaussian kernel here can be calculated as the outer product of two
	vectors with values populated from evaluating the 1D Gaussian PDF at each
	coordinate.
  """

  ############################
  ### TODO: YOUR CODE HERE ###

  k = cutoff_frequency * 4 + 1
  mean = k // 2
  sigma = cutoff_frequency
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

  # raise NotImplementedError('`create_Gaussian_kernel` function in '
  #   + '`student_code.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return kernel

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of shape (m, n, c)
  - filter: numpy nd-array of shape (k, j)
  Returns
  - filtered_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using OpenCV or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

  image_copy = image.copy()
  filtered_image = np.zeros(image.shape)

  filtered_image = image.copy()

  # Pad image
  padX = filter.shape[0]//2
  padY = filter.shape[1]//2
  image_copy = np.pad(filtered_image, ((padX, padX), (padY, padY), (0, 0)), 'reflect')

  # Pad filter
  # print("filter before", filter.shape)
  # filter = np.pad(filter, ((0, 0), (0, 0), (1, 1)), 'maximum')
  # print("filter after", filter.shape)

  # for m in range(image.shape[0]):
  # 	for n in range(image.shape[1]):
  # 		for c in range(image.shape[2]):
  # 			total = 0
  # 			for k in range(filter.shape[0]):
  # 				for j in range(filter.shape[1]):
  # 					g = filter[k][j]
  # 					first = np.clip(m+k, 0, image.shape[0]-1)
  # 					second = np.clip(n+j, 0, image.shape[1]-1)
  # 					f = image[first][second][c]
  # 					total += (g*f)
  # 			filtered_image[m][n][c] = total

  for m in range(image_copy.shape[0] - filter.shape[0]):
  	for n in range(image_copy.shape[1] - filter.shape[1]):
  		for c in range(image_copy.shape[2]):
  			aSlice = image_copy[m:m+filter.shape[0], n:n+filter.shape[1], c]
  			combined = np.multiply(aSlice, filter)
  			final = np.sum(combined)
  			filtered_image[m][n][c] = final
  # filtered_image = filtered_image[0:image_copy.shape[0], 0:image_copy.shape[1], 0:image_copy.shape[2]]

 #  raise NotImplementedError('`my_imfilter` function in `student_code.py` ' +
	# 'needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and a low-pass filter and creates a hybrid image. Returns
  the low frequency content of image1, the high frequency content of image 2,
  and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (x, y)
  Returns
  - low_frequencies: numpy nd-array of shape (m, n, c)
  - high_frequencies: numpy nd-array of shape (m, n, c)
  - hybrid_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
	frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values of the hybrid image are between
	0 and 1. This is known as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
	in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  assert filter.shape[0] <= image1.shape[0]
  assert filter.shape[1] <= image1.shape[1]
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter)
  hybrid_image = low_frequencies + high_frequencies
  hybrid_image = np.clip(hybrid_image, 0, 1)

 #  raise NotImplementedError('`create_hybrid_image` function in ' +
	# '`student_code.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
