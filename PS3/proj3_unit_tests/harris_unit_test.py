#!/usr/bin/python3

import numpy as np
import cv2
import pdb
import numpy as np
from pathlib import Path

from proj3_code.student_harris import (
  get_gaussian_kernel, 
  get_gradients,
  my_filter2D,
  remove_border_vals, 
  second_moments,
  corner_response,
  non_max_suppression, 
  get_interest_points
)

from proj3_code.utils import (
    load_image,
)

ROOT = Path(__file__).resolve().parent.parent  # ../..

def verify(function) -> str:
  """ Will indicate with a print statement whether assertions passed or failed
    within function argument call.

    Args:
    - function: Python function object

    Returns:
    - string
  """
  try:
    function()
    return "\x1b[32m\"Correct\"\x1b[0m"
  except AssertionError:
    return "\x1b[31m\"Wrong\"\x1b[0m"


def test_get_gradients():
  sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
  sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  dummy_image = np.arange(48).reshape(-1, 6).astype('float32')

  true_ix = cv2.filter2D(dummy_image, ddepth = -1, kernel = sobel_x, borderType = cv2.BORDER_CONSTANT)
  true_iy = cv2.filter2D(dummy_image, ddepth = -1, kernel = sobel_y, borderType = cv2.BORDER_CONSTANT)

  sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

  true_ix2 = cv2.filter2D(dummy_image, ddepth = -1, kernel = sobel_x, borderType = cv2.BORDER_CONSTANT)
  true_iy2 = cv2.filter2D(dummy_image, ddepth = -1, kernel = sobel_y, borderType = cv2.BORDER_CONSTANT)

  ix, iy = get_gradients(dummy_image)

  assert (np.allclose(true_ix, ix) and np.allclose(true_iy, iy)) or (np.allclose(true_ix2, ix) and np.allclose(true_iy2, iy))


def test_get_gradients2():

  sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
  sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  dummy_image = np.array(
    [
      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ])

  true_ix = cv2.filter2D(dummy_image, ddepth = -1, kernel = sobel_x, borderType = cv2.BORDER_CONSTANT)
  true_iy = cv2.filter2D(dummy_image, ddepth = -1, kernel = sobel_y, borderType = cv2.BORDER_CONSTANT)

  sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

  true_ix2 = cv2.filter2D(dummy_image, ddepth = -1, kernel = sobel_x, borderType = cv2.BORDER_CONSTANT)
  true_iy2 = cv2.filter2D(dummy_image, ddepth = -1, kernel = sobel_y, borderType = cv2.BORDER_CONSTANT)

  ix, iy = get_gradients(dummy_image)

  assert (np.allclose(true_ix, ix) and np.allclose(true_iy, iy)) or (np.allclose(true_ix2, ix) and np.allclose(true_iy2, iy))
  
  


def test_gaussian_kernel():
  gt = np.array([[0.11053241, 0.11139933, 0.11053241],
       [0.11139933, 0.11227304, 0.11139933],
       [0.11053241, 0.11139933, 0.11053241]])
  test = get_gaussian_kernel(3, 8)
  assert np.allclose(test.sum(), 1) and np.allclose(gt, get_gaussian_kernel(ksize=3, sigma = 8))

def test_second_moment():
  ix = np.array([[4., 3., 0.],[0., 3., 2.],[4., 2., 2.,]])
  iy = np.array([[ 2.,  2.,  0.],[2.,  1.,  0.],[3.,  2.,  1.]])

  #sanity check, ksize=1 sigma =1, output = input
  sx2, sy2, sxsy = second_moments(ix, iy, ksize=1, sigma=1)
  print("Expected:")
  print(ix*ix)
  print(iy*iy)
  print(ix*iy)
  print("Actual:")
  print(sx2)
  print(sy2)
  print(sxsy)
  assert np.allclose(sx2,ix * ix) and np.allclose(sy2, iy * iy) and np.allclose(sxsy, ix * iy)


  #case 2: ksize =3 sigma = 3
  sx2, sy2, sxsy = second_moments(ix, iy, ksize=3, sigma=3)
  out_sx2 = np.array([[3.8941, 4.3319, 2.4334],
        [6.0285, 6.8509, 3.3397],
        [3.3286, 4.1865, 2.3461]])

  out_sy2 = np.array([[1.4902, 1.4718, 0.5594],
        [2.9178, 2.9749, 1.0822],
        [2.0880, 2.1505, 0.6790]])

  out_sxsy = np.array([[1.9562, 1.9616, 0.9997],
        [3.6715, 3.8438, 1.6355],
        [2.2083, 2.4012, 1.0126]])
  print("Expected:")
  print(out_sx2)
  print(out_sy2)
  print(out_sxsy)
  print("Actual:")
  print(sx2)
  print(sy2)
  print(sxsy)
  assert np.allclose(out_sx2,sx2,rtol=1e-04)
  assert np.allclose(out_sy2,sy2,rtol=1e-04)
  assert np.allclose(out_sxsy,sxsy,rtol=1e-04)


def test_corner_response():
  """
  test CornerResponseLayer. Convert tensor of shape (1, 3, 3, 3) to (1, 1, 3, 3)
  """
  sx2 = np.array([[4, 3, 0],
      [0, 3, 2],  
      [4, 2, 2]])
  
  sy2 = np.array([[2, 2, 0],
      [2, 1, 0],
      [3, 2, 1]])
  
  sxsy = np.array([[3, 0, 3],
      [4, 4, 1],
      [2, 0, 1]])


  R = corner_response(sx2, sy2, sxsy, 0.05)
  R_gt = np.array(
      [[ -2.8000,   4.7500,  -9.0000],
      [-16.2000, -13.8000,  -1.2000],
      [  5.5500,   3.2000,   0.5500]]
    )
  assert np.allclose(R, R_gt,rtol=1e-04)



def test_get_interest_points():
  """
  Tests that get_interest_points function can get the correct coordinate. 
  """    
  dummy_image = np.array(
    [
      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ])

  x, y, R, confidence = get_interest_points(dummy_image)
  xy = [(x[i],y[i]) for i in range(len(x))]
  assert (9,9) in xy #(9,9) must be in the interest points



def test_find_single_valid_corner():

  img = np.array([
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])


  x, y, R, confidence = get_interest_points(img * 250)
  assert (x[0] == 16) and (y[0] == 10) and (np.max(R) != confidence[0])

