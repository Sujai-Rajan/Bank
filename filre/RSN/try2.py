import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob


# function to pre proces image
def preprocess_image(image):
    # Apply white balance
    wb = cv2.xphoto.createSimpleWB()
    image_balanced = wb.balanceWhite(image)

    # convert to gray scale
    gray_image = cv2.cvtColor(image_balanced, cv2.COLOR_BGR2GRAY)

    # apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # apply gaussian blur
    denoised_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    return denoised_image


# function for feature detection
def feature_detection(image):
    # create ORB object with desired number of features
    orb = cv2.ORB_create(nfeatures=1500)

    # find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


# function for feature matching
def match_features(descriptors1, descriptors2):
    # create a FLANN matcher with L2 norm and k=2
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match the features and filter good matches using Lowe's ratio test
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    return good_matches

# function for homography
def find_homography(keypoints1, keypoints2, matches):
    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate the homography matrix using RANSAC
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H

def create_mosaic(image_list):
    # Pre-process the images
    preprocessed_images = [preprocess_image(image) for image in image_list]

    # Detect keypoints and descriptors in the images
    keypoints_and_descriptors = [feature_detection(image) for image in preprocessed_images]

    # Initialize the final mosaic with the first image
    mosaic = image_list[0]

    for i in range(1, len(image_list)):
        # Match features between the current image and the mosaic
        matches = match_features(keypoints_and_descriptors[i - 1][1], keypoints_and_descriptors[i][1])

        # Estimate the homography matrix
        H = find_homography(keypoints_and_descriptors[i - 1][0], keypoints_and_descriptors[i][0], matches)

        # Stitch the current image onto the mosaic
        mosaic = cv2.warpPerspective(mosaic, np.dot(H, np.eye(3)), (mosaic.shape[1] + image_list[i].shape[1], mosaic.shape[0]))
        mosaic[0:image_list[i].shape[0], 0:image_list[i].shape[1]] = image_list[i]

    return mosaic


def main():
    # Load sample images from a directory
    image_dir = 'third'
    image_files = os.listdir(image_dir)
    image_list = [cv2.imread(os.path.join(image_dir, image_file)) for image_file in image_files]

    # Display the images
    for i, image in enumerate(image_list):
        plt.subplot(1, len(image_list), i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image {}'.format(i + 1))
        plt.axis('off')
        plt.show()

    # Create the image mosaic
    mosaic = create_mosaic(image_list)

    # Display the final mosaic
    plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    plt.title('Image Mosaic')
    plt.show()

if __name__ == '__main__':
    main()
