# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:13:16 2020

@author: nguyenrobot
# copyright nguyenrobot
# https://github.com/nguyenrobot/

"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
import math
#%matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 255, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def color_selection(frame, RGB_thd):
# Define color selection threshold
# Example : to keep white and yellow RGB_thd should be
# RGB_thd = [[200, 200, 200], [200, 200, 0]]

    color_selection_ind = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
    result              = np.copy(frame)
    
    # Define selection by color / below the threshold
    for RGB_thd_i in RGB_thd:
        color_selection_ind[:,:] = ((result[:,:,0] > RGB_thd_i[0]) & \
                                    (result[:,:,1] > RGB_thd_i[1]) & \
                                    (result[:,:,2] > RGB_thd_i[2])) \
                                    | color_selection_ind[:,:]
    result[~color_selection_ind] = [0, 0, 0]
    #result                     = np.copy(result[color_selection_ind])
    return result, color_selection_ind

image = img.imread('test_images/solidYellowLeft.jpg')
plt.imshow(image)

# NOTE: The output you return should be a color image (3 channel) for processing video below
# TODO: put your pipeline here,
# you should return the final output (image where lines are drawn on lanes)
result = image

result = grayscale(result)
plt.figure()
plt.imshow(result, cmap='gray')

kernel_size = 5
result = gaussian_blur(result, kernel_size)
plt.figure()
plt.imshow(result, cmap='gray')

low_threshold   = 5
high_threshold  = 100
result          = canny(result, low_threshold, high_threshold)
plt.figure()
plt.imshow(result, cmap='gray')

vertices        = np.array([[(0,np.int(image.shape[0]/2) + 100), \
                      (0,image.shape[0]-1), \
                      (image.shape[1]-1,image.shape[0]-1), \
                      (image.shape[1]-1,np.int(image.shape[0]/2) + 100), \
                      (np.int(image.shape[1]/2) + 200, np.int(image.shape[0]/2) + 50), \
                      (np.int(image.shape[1]/2) - 200, np.int(image.shape[0]/2) + 50)]], \
                    dtype=np.int32)
zone_interest   = np.copy(image)
cv2.fillPoly(zone_interest, vertices, [255, 255, 255])
plt.figure()
plt.imshow(zone_interest, cmap='gray')

result          = region_of_interest(result, vertices)
plt.figure()
plt.imshow(result, cmap='gray')

minLineLength   = 50
maxLineGap      = 100
rho             = 1
theta           = np.pi/180
result, lines   = hough_lines(result, rho, theta, 100, minLineLength, maxLineGap)
plt.figure()
plt.imshow(result, cmap='gray')

result          = weighted_img(result, image)
plt.figure()
plt.imshow(result, cmap='gray')
plt.imsave('frame_weighted_img.jpg', result)