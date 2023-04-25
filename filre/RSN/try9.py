import cv2
import numpy as np

def enhance_image(image):
    # Convert image to YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Stretch the histogram of Y channel
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])

    # Convert the image back to BGR color space
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # White balancing using Simplest Color Balance algorithm
    def simplest_cb(img, percent=1):
        assert img.shape[2] == 3
        assert 0 < percent <= 100
        half_percent = percent / 200.0
        channels = cv2.split(img)
        out_channels = []
        for channel in channels:
            assert len(channel.shape) == 2
            flat = channel.flatten()
            flat = np.sort(flat)
            n_pixels = flat.shape[0]
            low_val = flat[int(n_pixels * half_percent)]
            high_val = flat[int(n_pixels * (1 - half_percent))]
            channel = cv2.normalize(channel, None, low_val, high_val, cv2.NORM_MINMAX)
            out_channels.append(channel)
        return cv2.merge(out_channels)

    result = simplest_cb(result, 1)
    return result


# read the image
img1 = cv2.imread('third/3_2.tif')
img2 = cv2.imread('third/3_3.tif')

# Display the images
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

# Enhance the images
img1 = enhance_image(img1)
img2 = enhance_image(img2)

# Display the enhanced images
cv2.imshow('img1_enhanced', img1)
cv2.imshow('img2_enhanced', img2)

# apply preprocessing
# morphological opening
kernel = np.ones((5,5),np.uint8)
img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)


# Display the preprocessed images
cv2.imshow('img1_preprocessed', img1)
cv2.imshow('img2_preprocessed', img2)

cv2.waitKey(0)
