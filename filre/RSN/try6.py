import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import tqdm.notebook as tqdm

class ImageStitcher:
    def __init__(self):
        pass

    def read_image(self, path):
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_gray, img, img_rgb

    def SIFT(self, img):
        # siftDetector = cv2.xfeatures2d.SIFT_create(2500)
        # kp, des = siftDetector.detectAndCompute(img, None)
        # Use ORB instead of SIFT
        orb = cv2.ORB_create(10000)
        kp, des = orb.detectAndCompute(img, None)
        return kp, des
        

    def matcher(self, kp1, des1, img1, kp2, des2, img2, threshold):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # draw the matches on the image good and all matches
        img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
        plt.imshow(img4)
        plt.show()

        good = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append([m])

        matches = []
        for pair in good:
            matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

        matches = np.array(matches)


        # draw the matches on the image good and all matches
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
        plt.imshow(img3)
        plt.show()

        return matches

    def homography(self, pairs):
        rows = []
        for i in range(pairs.shape[0]):
            p1 = np.append(pairs[i][0:2], 1)
            p2 = np.append(pairs[i][2:4], 1)
            row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]]
            row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]]
            rows.append(row1)
            rows.append(row2)
        rows = np.array(rows)
        U, s, V = np.linalg.svd(rows)
        H = V[-1].reshape(3, 3)
        H = H / H[2, 2]
        return H

    def random_point(self, matches, k=4):
        idx = random.sample(range(len(matches)), k)
        point = [matches[i] for i in idx]
        return np.array(point)

    def get_error(self, points, H):
        num_points = len(points)
        all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
        all_p2 = points[:, 2:4]
        estimate_p2 = np.zeros((num_points, 2))
        for i in range(num_points):
            temp = np.dot(H, all_p1[i])
            estimate_p2[i] = (temp / temp[2])[0:2]
        errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2

        return errors
    

    def ransac(self, matches, threshold, iters):
        num_best_inliers = 0

        for i in range(iters):
            points = self.random_point(matches)
            H = self.homography(points)

            if np.linalg.matrix_rank(H) < 3:
                continue

            errors = self.get_error(matches, H)
            idx = np.where(errors < threshold)[0]
            inliers = matches[idx]

            num_inliers = len(inliers)
            if num_inliers > num_best_inliers:
                best_inliers = inliers.copy()
                num_best_inliers = num_inliers
                best_H = H.copy()

        print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
        return best_inliers, best_H

    # def stitch_img(self, left, right, H):
    #     left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    #     right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    #     height_l, width_l, channel_l = left.shape
    #     corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    #     corners_new = [np.dot(H, corner) for corner in corners]
    #     corners_new = np.array(corners_new).T
    #     x_news = corners_new[0] / corners_new[2]
    #     y_news = corners_new[1] / corners_new[2]
    #     y_min = min(y_news)
    #     x_min = min(x_news)

    #     translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    #     H = np.dot(translation_mat, H)

    #     height_new = int(round(abs(y_min) + height_l))
    #     width_new = int(round(abs(x_min) + width_l))
    #     size = (width_new, height_new)

    #     warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
    #     warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

    #     result = cv2.addWeighted(warped_l, 0.5, warped_r, 0.5, 0)
        
    #     # Convert the stitched image to uint8
    #     stitch_image_uint8 = (result * 255).astype(np.uint8)
    #     return stitch_image_uint8

    def stitch_img(self, left, right, H):
        print("Stitching Images...")

        left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        height_l, width_l, channel_l = left.shape
        corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
        corners_new = [np.dot(H, corner) for corner in corners]
        corners_new = np.array(corners_new).T 
        x_news = corners_new[0] / corners_new[2]
        y_news = corners_new[1] / corners_new[2]
        y_min = min(y_news)
        x_min = min(x_news)

        translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        H = np.dot(translation_mat, H)

        height_new = int(round(abs(y_min) + height_l))
        width_new = int(round(abs(x_min) + width_l))
        size = (width_new, height_new)

        warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

        height_r, width_r, channel_r = right.shape
        height_new = int(round(abs(y_min) + height_r))
        width_new = int(round(abs(x_min) + width_r))
        size = (width_new, height_new)

        warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

        stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]

        # Find the minimum dimensions between warped_l and warped_r
        min_height = min(warped_l.shape[0], warped_r.shape[0])
        min_width = min(warped_l.shape[1], warped_r.shape[1])

        black = np.zeros(3)  # Black pixel.
        
        for i in range(min_height):
            for j in range(min_width):
                pixel_l = warped_l[i, j, :]
                pixel_r = warped_r[i, j, :]
                
                if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                    warped_l[i, j, :] = pixel_l
                elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                    warped_l[i, j, :] = pixel_r
                elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                    warped_l[i, j, :] = (pixel_l + pixel_r) / 2
                else:
                    pass
                  

        # Convert the stitched image to uint8
        stitch_image_uint8 = (stitch_image * 255).astype(np.uint8)


        print("Stitching Done!")
        return stitch_image_uint8


    # def stitch_images(self, folder):
    #     file_list = [f for f in os.listdir(folder) if f.endswith('.tif')]
    #     file_list.sort()
    #     z=0
    #     for i in range(0, len(file_list), 2):
    #         left_gray, left_rgb, _ = self.read_image(os.path.join(folder, file_list[i]))
    #         right_gray, right_rgb, _ = self.read_image(os.path.join(folder, file_list[i + 1]))

    #         kp_left, des_left = self.SIFT(left_gray)
    #         kp_right, des_right = self.SIFT(right_gray)

    #         matches = self.matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)
    #         inliers, H = self.ransac(matches, 0.5, 2000)

    #         stitched_image = self.stitch_img(left_rgb, right_rgb, H)
    #         # figure = plt.figure()
    #         # plt.imshow(stitched_image)
    #         # plt.show()
    #         z +=1
    #         cv2.imshow(f"Stitched Image {z}", stitched_image)


    # def stitch_images(self, folder):
    #     file_list = [f for f in os.listdir(folder) if f.endswith('.tif')]
    #     file_list.sort()

    #     # Read the first two images and stitch them together
    #     left_gray, left_rgb, _ = self.read_image(os.path.join(folder, file_list[0]))
    #     right_gray, right_rgb, _ = self.read_image(os.path.join(folder, file_list[1]))

    #     kp_left, des_left = self.SIFT(left_gray)
    #     kp_right, des_right = self.SIFT(right_gray)

    #     matches = self.matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)
    #     inliers, H = self.ransac(matches, 0.5, 2000)

    #     stitched_image = self.stitch_img(left_rgb, right_rgb, H)

    #     # Stitch the third image with the previously stitched image
    #     left_gray, left_rgb, _ = self.read_image(os.path.join(folder, file_list[2]))

    #     kp_left, des_left = self.SIFT(stitched_image)
    #     kp_right, des_right = self.SIFT(left_gray)

    #     matches = self.matcher(kp_left, des_left, stitched_image, kp_right, des_right, left_rgb, 0.5)
    #     inliers, H = self.ransac(matches, 0.5, 2000)

    #     final_stitched_image = self.stitch_img(stitched_image, left_rgb, H)

    #     cv2.imshow("Final Stitched Image", final_stitched_image)


    def stitch_images(self, folder):
        file_list = [f for f in os.listdir(folder) if f.endswith('.tif')]
        file_list.sort()

        # Read the first image as the initial stitched image
        _, stitched_image, _ = self.read_image(os.path.join(folder, file_list[0]))

        # Iterate through the rest of the images and stitch them with the current stitched image
        for i in range(1, len(file_list)):
            right_gray, right_rgb, _ = self.read_image(os.path.join(folder, file_list[i]))

            kp_left, des_left = self.SIFT(stitched_image)
            kp_right, des_right = self.SIFT(right_gray)

            matches = self.matcher(kp_left, des_left, stitched_image, kp_right, des_right, right_rgb, 0.6)



            inliers, H = self.ransac(matches, 0.6, 5000)


            stitched_image = self.stitch_img(stitched_image, right_rgb, H)

            cv2.imshow(f"Stitched Image Set-{i}", stitched_image)
            cv2.waitKey(0)

        cv2.imshow("Final Stitched Image", stitched_image)



def main():
    plt.rcParams['figure.figsize'] = [15, 15]
    stitcher = ImageStitcher()
    stitcher.stitch_images('third')

    cv2.waitKey(0)

if __name__ == '__main__':
    main()