import cv2
import numpy as np
import os

def preprocess_image(image):
    # Convert to grayscale if the image is not already grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Apply histogram equalization to improve contrast
    image = cv2.equalizeHist(image)
    
    # # Apply noise reduction using a Gaussian filter
    # image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply bilateral filter to reduce noise while keeping edges sharp 
    image = cv2.bilateralFilter(image, 9, 75, 75)


    
    return image


def detect_and_compute(image):
    # Initialize the SIFT detector
    sift = cv2.SIFT_create(5000)
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # display the keypoints and descriptors on the image
    image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    return keypoints, descriptors


# def match_features(desc1, desc2):
#     # Initialize the FLANN matcher
#     index_params = dict(algorithm=0, trees=5)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)

#     # Match features
#     matches = flann.knnMatch(desc1, desc2, k=2)

#     # Apply ratio test to filter good matches
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good_matches.append(m)

#     return good_matches

def match_features(desc1, desc2):
    # Initialize the Brute-Force matcher with cross-check
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match features
    matches = bf.match(desc1, desc2)

    # Sort matches by distance (ascending)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches[:50]


def find_homography(kp1, kp2, matches):
    # Extract the matched keypoints' coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate homography using RANSAC
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # display the number of inliers
    print(f'Number of inliers: {np.sum(mask)}')


    return homography, mask


# def stitch_images(image1, image2, homography, mask):
#     # Warp image1 to image2's perspective
#     h1, w1 = image1.shape[:2]
#     h2, w2 = image2.shape[:2]

#     # Transform image1's corners using homography
#     corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])
#     corners2 = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2), homography)

#     # Compute the offset required to shift image1 to the right and down
#     x_offset, y_offset = np.min(corners2, axis=0).ravel()

#     # Update homography matrix to shift image1 to the right and down
#     homography[0, 2] -= x_offset
#     homography[1, 2] -= y_offset

#     # Warp image1 to image2's perspective
#     warped_image1 = cv2.warpPerspective(image1, homography, (w2, h2))

#     # Blend images using multi-band blending
#     levels = 6

#     # Create Gaussian pyramid
#     gaussian_pyramid = []
#     gaussian_pyramid.append(warped_image1.copy())
#     for i in range(levels - 1):
#         gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[-1]))

#     # Create Laplacian pyramid
#     laplacian_pyramid = []
#     laplacian_pyramid.append(gaussian_pyramid[-1])
#     for i in range(levels - 1, 0, -1):
#         size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
#         gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
#         laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
#         laplacian_pyramid.append(laplacian)
#     laplacian_pyramid.reverse()

#     # Create mask pyramid
#     mask_pyramid = []
#     mask_pyramid.append(np.float32(np.ones_like(warped_image1)))
#     for i in range(levels - 1):
#         mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))

#     # Blend Laplacian pyramid using mask pyramid
#     blended_pyramid = []
#     for i in range(levels):
#         blended_pyramid.append(laplacian_pyramid[i] * mask_pyramid[i] + (1 - mask_pyramid[i]) * gaussian_pyramid[i])

#     # Reconstruct blended image from blended Laplacian pyramid
#     blended_image = blended_pyramid[-1]
#     for i in range(levels - 2, -1, -1):
#         size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
#         blended_image = cv2.pyrUp(blended_image, dstsize=size)
#         blended_image = cv2.add(blended_image, blended_pyramid[i])

#     # Combine image2 and blended image
#     result = np.zeros((max(h2 + int(y_offset), h1), max(w2 + int(x_offset), w1), 3), dtype=np.uint8)
#     result[y_offset:y_offset+h2, x_offset:x_offset+w2] = image2
#     result[:h1, :w1] = blended_image

#     return result







def stitch_images(image1, image2, homography):
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    
    # display image adn attributes
    cv2.imshow("image1", image1)
    cv2.imshow("image2", image2)
    print("image1.shape: ", image1.shape)
    print("image2.shape: ", image2.shape)
    cv2.waitKey(0)


    # Convert grayscale images to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    # display_image
    cv2.imshow("image1_rgb", image1_rgb)
    cv2.imshow("image2_rgb", image2_rgb)
    print("image1_rgb.shape: ", image1_rgb.shape)
    print("image2_rgb.shape: ", image2_rgb.shape)
    cv2.waitKey(0)

    
    # Warp the first image
    h1, w1 = image1_rgb.shape[:2]
    warped_img1 = image1_rgb.copy()

    # Display image and all attributes
    cv2.imshow("warped_img1", warped_img1)
    print("warped_img1.shape: ", warped_img1.shape)
    cv2.waitKey(0)


    
    # Warp the second image
    h2, w2 = image2_rgb.shape[:2]
    warped_img2 = cv2.warpPerspective(image2_rgb, homography, (w1 + w2, h1+ h2))

    # Display image and all attributes
    cv2.imshow("warped_img2", warped_img2)
    print("warped_img2.shape: ", warped_img2.shape)
    cv2.waitKey(0)

    
    
    # Find the seam mask
    seam_mask = np.zeros(((h1+ h2), w1 + w2), dtype=np.uint8)
    seam_mask[0:h1, 0:w1] = 1
    seam_mask[0:h2, w1:w1 + w2] = 1

    # Display image and all attributes
    cv2.imshow("seam_mask", seam_mask)
    print("seam_mask.shape: ", seam_mask.shape)
    cv2.waitKey(0)


    
    # Compute the stitched image using the stitcher
    status, stitched_image = stitcher.stitch((warped_img1, warped_img2), seam_mask)

    # status, stitched_image = stitcher.stitch((warped_img1, warped_img2))

    # Display image and all attributes
    cv2.imshow("stitched_image", stitched_image)
    print("stitched_image.shape: ", stitched_image.shape)
    cv2.waitKey(0)


    
    # Check if the stitched image is empty
    if status != cv2.Stitcher_OK:
        print("Error during stitching. Status code:", str(status))
        return None
    
    # Convert the stitched image back to grayscale
    stitched_image_gray = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2GRAY)
    
    return stitched_image_gray







# def stitch_images(image1, image2, homography):
#     # Get the dimensions of the input images
#     h1, w1 = image1.shape
#     h2, w2 = image2.shape

#     # Calculate the size of the output mosaic
#     corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
#     corners2 = cv2.perspectiveTransform(corners1, homography)
#     min_corners = np.int32(np.min(corners2, axis=0).ravel())
#     max_corners = np.int32(np.max(corners2, axis=0).ravel())
#     x_offset, y_offset = -min_corners
#     mosaic_w = max(w1, max_corners[0]) - min_corners[0]
#     mosaic_h = max(h1, max_corners[1]) - min_corners[1]

#     # Initialize an empty mosaic image with the calculated size
#     mosaic = np.zeros((mosaic_h, mosaic_w), dtype=np.uint8)

#     # Translate the homography matrix to account for the offsets
#     translation_homography = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]], dtype=np.float32)
#     homography = np.dot(translation_homography, homography)

#     # Warp image2 onto the mosaic using the translated homography matrix
#     cv2.warpPerspective(image2, homography, (mosaic_w, mosaic_h), mosaic, borderMode=cv2.BORDER_TRANSPARENT)

#     # Blend image1 into the mosaic
#     mask = np.zeros((mosaic_h, mosaic_w), dtype=np.uint8)
#     mask[y_offset:y_offset+h1, x_offset:x_offset+w1] = 255
#     mosaic = cv2.addWeighted(mosaic, 0.5, cv2.warpPerspective(image1, translation_homography, (mosaic_w, mosaic_h)), 0.5, 0, dst=mosaic, mask=mask)

#     return mosaic






def main():
    # Load images from a folder

    image_folder = "test1/"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
    images = [cv2.imread(os.path.join(image_folder, f)) for f in image_files]

    # print names
    print(image_files)

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

        print("Number of matches: ", len(matches))

        # display the matches on the images in a vertical stack
        cv2.imshow("Matches", cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(0, 255, 0), singlePointColor=(0, 255, 0)))
        cv2.waitKey(0)


        # Estimate homography
        homography, mask = find_homography(kp1, kp2, matches)
        
        # display the matches on the images and wait for a keypress
        cv2.imshow("Homography", cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(0, 255, 0), singlePointColor=(0, 255, 0), matchesMask=mask.ravel().tolist()))
        cv2.imshow("mask", mask)
        cv2.waitKey(0)

        # Check if there are enough inliers
        if np.sum(mask) > 10:
            # Stitch images
            stitched_image = stitch_images(img1, img2, homography)
        else:
            print(f"Skipping image {i} due to insufficient inliers.")

            # print shape
            print(stitched_image.shape)

            # Show result
            cv2.imshow("Mosaic", stitched_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
