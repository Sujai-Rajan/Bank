# import cv2

# # Read the image
# img = cv2.imread('third/3_2.tif')

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply median filtering with a kernel size of 3
# filtered_img = cv2.medianBlur(gray_img, 3)

# # applying gaussian blur to main image
# gaussian_img = cv2.GaussianBlur(gray_img,(5,5),0)

# # applying bilateral filter to main image
# bilateral_img = cv2.bilateralFilter(gray_img,9,100,100)

# # apply adaptive thresholding
# thresh = cv2.adaptiveThreshold(gaussian_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# # Display the original and filtered images
# cv2.imshow('Original Image', gray_img)
# cv2.imshow('Filtered Image', filtered_img)
# cv2.imshow('Gaussian Image', gaussian_img)
# cv2.imshow('Bilateral Image', bilateral_img)
# cv2.imshow('Adaptive Thresholding', thresh)
# cv2.waitKey(0)

import cv2
import pywt
import numpy as np

# Read the input image
img = cv2.imread('third/3_2.tif')



# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply wavelet denoising with 'db4' wavelet and soft thresholding
coeffs = pywt.wavedec2(img, wavelet='db4', mode='symmetric', level=5)
threshold = 0.05
coeffs_thresh = [pywt.threshold(i, value=threshold, mode='soft') for i in coeffs]

# Convert any lists to numpy arrays
coeffs_thresh = [np.array(i) if type(i) == list else i for i in coeffs_thresh]

# Convert the thresholded coefficients to the required format
coeffs_thresh = pywt.array_to_coeffs(coeffs_thresh, coeffs[0].shape, output_format='wavedec2')

# Reconstruct the denoised image
denoised_img = pywt.waverec2(coeffs_thresh, 'db4', mode='symmetric')

# Display the original and denoised images
cv2.imshow('Original Image', gray_img)
cv2.imshow('Denoised Image', denoised_img)
cv2.waitKey(0)




# import cv2
# import pywt
# import numpy as np

# # Read the image
# img = cv2.imread('third/3_2.tif')

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply wavelet denoising with 'db4' wavelet and soft thresholding
# coeffs = pywt.wavedec2(gray_img, wavelet='db4', mode='symmetric', level=5)
# threshold = 0.05
# coeffs_thresh = [pywt.threshold(i, value=threshold, mode='soft') for i in coeffs]

# # Convert any lists to numpy arrays
# coeffs_thresh = [np.array(i) if type(i) == list else i for i in coeffs_thresh]

# # Convert the thresholded coefficients to the required format
# coeffs_thresh = pywt.array_to_coeffs(coeffs_thresh, coeffs[0].shape, output_format='wavedec2')

# # Reconstruct the denoised image
# denoised_img = pywt.waverec2(coeffs_thresh, 'db4', mode='symmetric')

# # Display the original and denoised images
# cv2.imshow('Original Image', gray_img)
# cv2.imshow('Denoised Image', denoised_img)
# cv2.waitKey(0)
