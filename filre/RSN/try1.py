import os
import glob
import cv2
import numpy as np


def read_images_from_folder(folder_path):
    images = []
    img_paths = sorted(glob.glob(os.path.join(folder_path, '*.tif')))
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images


def detect_and_describe(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def find_homography(keypoints1, keypoints2, matches):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H


def laplacian_pyramid_blending(img1, img2, mask):
    # Generate Gaussian pyramids
    gaussian_img1 = [img1.copy()]
    gaussian_img2 = [img2.copy()]
    gaussian_mask = [mask.copy()]
    
    for i in range(6):
        img1_down = cv2.pyrDown(gaussian_img1[-1])
        gaussian_img1.append(img1_down)
        
        img2_down = cv2.pyrDown(gaussian_img2[-1])
        gaussian_img2.append(img2_down)
        
        mask_down = cv2.pyrDown(gaussian_mask[-1])
        gaussian_mask.append(mask_down)

    # Generate Laplacian pyramids
    laplacian_img1 = [gaussian_img1[-1]]
    laplacian_img2 = [gaussian_img2[-1]]
    
    for i in range(6, 0, -1):
        img1_expanded = cv2.pyrUp(gaussian_img1[i])
        laplacian_img1.append(cv2.subtract(gaussian_img1[i - 1], img1_expanded))
        
        img2_expanded = cv2.pyrUp(gaussian_img2[i])
        laplacian_img2.append(cv2.subtract(gaussian_img2[i - 1], img2_expanded))

    # Blend Laplacian pyramids
    blended_pyramid = []
    for lap_img1, lap_img2, gauss_mask in zip(laplacian_img1, laplacian_img2, gaussian_mask):
        blended_pyramid.append(cv2.addWeighted(lap_img1, gauss_mask / 255.0, lap_img2, 1 - gauss_mask / 255.0, 0))

    # Reconstruct the blended image from the blended pyramid
    blended_img = blended_pyramid[0]
    for i in range(1, 7):
        blended_img = cv2.pyrUp(blended_img)
        blended_img = cv2.add(blended_img, blended_pyramid[i])

    return blended_img

def stitch_images_vertically(img1, img2, H):
    img1_shape = img1.shape
    img2_shape = img2.shape

    # Calculate the vertical translation
    vertical_translation = int(np.abs(H[1, 2]))

    # Create an empty result image with enough space for the stitched images
    img_result = np.zeros((img1_shape[0] + vertical_translation, img1_shape[1]), dtype=np.uint8)

    # Place img1 in the result image
    img_result[:img1_shape[0], :img1_shape[1]] = img1

    # Apply the homography transformation to img2
    img2_transformed = cv2.warpPerspective(img2, H, (img1_shape[1], img1_shape[0] + vertical_translation))

    # Create a mask for img2_transformed
    mask = np.zeros((img1_shape[0] + vertical_translation, img1_shape[1]), dtype=np.uint8)
    mask[vertical_translation:vertical_translation + img2_shape[0], :img2_shape[1]] = 255
    mask = cv2.warpPerspective(mask, H, (img1_shape[1], img1_shape[0] + vertical_translation))

    # Blend the overlapping regions using Laplacian pyramid blending
    img_result = laplacian_pyramid_blending(img_result, img2_transformed, mask)
    
    return img_result


def main():
    folder_path = "third"

    images = read_images_from_folder(folder_path)

    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]

        keypoints1, descriptors1 = detect_and_describe(img1)
        keypoints2, descriptors2 = detect_and_describe(img2)

        matches = match_features(descriptors1, descriptors2)
        H = find_homography(keypoints1, keypoints2, matches)

        result = stitch_images_vertically(img1, img2, H)

        # Save the result image
        cv2.imwrite(f"stitched_{i+1}.jpg", result)

        # Update images list for the next iteration
        images[i + 1] = result


if __name__ == "__main__":
    main()