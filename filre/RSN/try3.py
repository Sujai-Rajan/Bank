import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale if the image is not already grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Apply histogram equalization to improve contrast
    image = cv2.equalizeHist(image)
    
    # Apply noise reduction using a Gaussian filter
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image


def detect_and_compute(image):
    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures=1000)  # You can adjust the number of features (1000) as needed

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def match_features(desc1, desc2):
    # Initialize the Brute-Force Hamming matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match features
    matches = matcher.match(desc1, desc2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def find_homography(kp1, kp2, matches):
    # Extract the matched keypoints' coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate homography using RANSAC
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return homography


def stitch_images(image1, image2, homography):
    # Compute the size of the stitched image
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, homography)
    min_x, min_y = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
    max_x, max_y = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)
    translation = np.float32([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    final_homography = translation @ homography
    stitched_image = cv2.warpPerspective(image1, final_homography, (max_x - min_x, max_y - min_y))
    stitched_image[-min_y:h2 - min_y, -min_x:w2 - min_x] = image2

    # Create Laplacian pyramids for both images
    lp1 = laplacian_pyramid(image1)
    lp2 = laplacian_pyramid(image2)

    # Create a mask for image1 and generate a Gaussian pyramid
    mask = np.zeros_like(image1, dtype=np.float32)
    mask[-min_y:h2 - min_y, -min_x:w2 - min_x] = 1
    gp_mask = gaussian_pyramid(mask)

    # Blend the Laplacian pyramids using the Gaussian pyramid mask
    blended_pyramid = []
    for l1, l2, gm in zip(lp1, lp2, gp_mask):
        blended_pyramid.append(l1 * gm + l2 * (1 - gm))

    # Reconstruct the blended image from the blended Laplacian pyramid
    blended_image = blended_pyramid[-1]
    for i in range(len(blended_pyramid) - 2, -1, -1):
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(blended_image, dstsize=size)
        blended_image = cv2.add(upsampled, blended_pyramid[i])

    return blended_image

def laplacian_pyramid(image, levels=6):
    gaussian_pyramid = [image]
    for i in range(levels - 1):
        down = cv2.pyrDown(gaussian_pyramid[-1]) 
        gaussian_pyramid.append(down)

    laplacian_pyramid = []
    for i in range(levels - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid

def gaussian_pyramid(image, levels=6):
    gaussian_pyramid = [image]
    for i in range(levels - 1):
        down = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(down)

    return gaussian_pyramid


def main():
    # Load images from a folder
    import os
    image_folder = "third/"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
    images = [cv2.imread(os.path.join(image_folder, f)) for f in image_files]

    # Pre-process images
    preprocessed_images = [preprocess_image(img) for img in images]

    # Detect keypoints and compute descriptors
    keypoints_desc = [detect_and_compute(img) for img in preprocessed_images]

    # Pairwise stitching
    stitched_image = preprocessed_images[0]
    for i in range(1, len(preprocessed_images)):
        img1, img2 = stitched_image, preprocessed_images[i]
        kp1, desc1 = keypoints_desc[i - 1]
        kp2, desc2 = keypoints_desc[i]

        # Match features
        matches = match_features(desc1, desc2)

        # Estimate homography
        homography = find_homography(kp1, kp2, matches)

        # Stitch images
        stitched_image = stitch_images(img1, img2, homography)

    # Show result
    cv2.imshow("Mosaic", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
