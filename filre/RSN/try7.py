# SIFT and Homography



import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import tqdm.notebook as tqdm

class ImageStitcher:
    def __init__(self):
        pass

    # function to preprocess the image for better feature detection and matching
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

    def read_image(self, path):
        img = cv2.imread(path)
        img_rgb = img
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # function to preprocess the image for better feature detection and matching
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
        img = enhance_image(img)






        return img_gray, img, img_rgb
    


    def SIFT(self, img):

        # siftDetector = cv2.xfeatures2d.SIFT_create(5000)
        siftDetector = cv2.xfeatures2d.SIFT_create(nfeatures=5000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        kp, des = siftDetector.detectAndCompute(img, None)
        return kp, des
        
    def ORB(self, img):
        orbDetector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        kp, des = orbDetector.detectAndCompute(img, None)
        return kp, des

    def AKAZE(self, img):
        # akazeDetector = cv2.AKAZE_create()
        akazeDetector = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)
        kp, des = akazeDetector.detectAndCompute(img, None)
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
    

    def least_squares(self, matches):
        rows = []
        for i in range(matches.shape[0]):
            p1 = np.append(matches[i][0:2], 1)
            p2 = np.append(matches[i][2:4], 1)
            row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]]
            row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]]
            rows.append(row1)
            rows.append(row2)
        rows = np.array(rows)
        U, s, V = np.linalg.svd(rows)
        H = V[-1].reshape(3, 3)
        H = H / H[2, 2]
        return H


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
    

    def plot_keypoints_histogram(self, file_list, folder):
        num_keypoints = []
        for file in file_list:
            gray_img, _, _ = self.read_image(os.path.join(folder, file))
            kp, _ = self.SIFT(gray_img)
            num_keypoints.append(len(kp))

        fig, ax = plt.subplots()
        x = np.arange(len(file_list))
        width = 0.35

        rects1 = ax.bar(x, num_keypoints, width)

        ax.set_xlabel("Image")
        ax.set_ylabel("Number of Keypoints")
        ax.set_title("Keypoints Detected for Each Image")
        ax.set_xticks(x)
        ax.set_xticklabels(file_list, rotation='vertical')

        # Add values on top of the bars
        for i, v in enumerate(num_keypoints):
            ax.text(i, v, str(v), ha='center')

        plt.show()


    def plot_matches_histogram(self, file_list, folder, threshold):
        all_matches = []
        good_matches = []
        for i in range(len(file_list) - 1):
            gray_img1, _, _ = self.read_image(os.path.join(folder, file_list[i]))
            gray_img2, _, _ = self.read_image(os.path.join(folder, file_list[i + 1]))
            kp1, des1 = self.SIFT(gray_img1)
            kp2, des2 = self.SIFT(gray_img2)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=3)

            all_matches.append(len(matches))

            good = []
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good.append([m])

            good_matches.append(len(good))

        fig, ax = plt.subplots()
        x = np.arange(len(file_list) - 1)
        width = 0.35

        rects1 = ax.bar(x - width / 2, all_matches, width, label='All Matches')
        rects2 = ax.bar(x + width / 2, good_matches, width, label='Good Matches')

        ax.set_xlabel("Image Pairs")
        ax.set_ylabel("Number of Matches")
        ax.set_title("Matches Before and After Ratio Test")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}-{i + 1}" for i in range(1, len(file_list))], rotation='vertical')
        ax.legend()

        # Add values on top of the bars
        for i, v in enumerate(all_matches):
            ax.text(i - width / 2, v, str(v), ha='center')
        for i, v in enumerate(good_matches):
            ax.text(i + width / 2, v, str(v), ha='center')

        plt.show()


    def plot_inliers_outliers(self, file_list, folder, threshold, ransac_iters):
        inliers_count = []
        outliers_count = []
        total_matches_count = []
        for i in range(len(file_list) - 1):
            gray_img1, _, _ = self.read_image(os.path.join(folder, file_list[i]))
            gray_img2, _, _ = self.read_image(os.path.join(folder, file_list[i + 1]))
            kp1, des1 = self.SIFT(gray_img1)
            kp2, des2 = self.SIFT(gray_img2)
            matches = self.matcher(kp1, des1, gray_img1, kp2, des2, gray_img2, 0.5)
            inliers, _ = self.ransac(matches, threshold, ransac_iters)

            inliers_count.append(len(inliers))
            outliers_count.append(len(matches) - len(inliers))
            total_matches_count.append(len(matches))

        fig, ax = plt.subplots()
        x = np.arange(len(file_list) - 1)
        width = 0.25

        rects1 = ax.bar(x - width, total_matches_count, width, label='Total Matches')
        rects2 = ax.bar(x, inliers_count, width, label='Inliers')
        rects3 = ax.bar(x + width, outliers_count, width, label='Outliers')

        ax.set_xlabel("Image Pairs")
        ax.set_ylabel("Number of Points")
        ax.set_title("Total Matches, Inliers, and Outliers After RANSAC")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}-{i + 1}" for i in range(1, len(file_list))], rotation='vertical')
        ax.legend()

        # Add annotations to the bars
        for rect in rects1 + rects2 + rects3:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.show()

    def plot_comparison(self, ransac_errors, least_squares_errors, threshold):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # increase size of markers
        ax[0].scatter(range(len(ransac_errors)), ransac_errors, c='green', label='Inliers')
        ax[0].scatter(range(len(ransac_errors)), ransac_errors, c=np.where(ransac_errors < threshold, 'blue', 'red'), label='Outliers\nblue if <threshold\nred if >threshold', s=10)
        ax[0].set_title('RANSAC Model Fitting')
        ax[0].set_xlabel('Matches')
        ax[0].set_ylabel('Squared Error')
        ax[0].legend()

        ax[1].scatter(range(len(least_squares_errors)), least_squares_errors, c='green', label='Inliers')
        ax[1].scatter(range(len(least_squares_errors)), least_squares_errors, c=np.where(least_squares_errors < threshold, 'blue', 'red'), label='Outliers\nblue if <threshold\nred if >threshold', s=10)
        ax[1].set_title('Least Squares Model Fitting')
        ax[1].set_xlabel('Matches')
        ax[1].set_ylabel('Squared Error')
        ax[1].legend()
        plt.show()


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


    def stitch_images(self, folder, feature_detector):
        file_list = [f for f in os.listdir(folder) if f.endswith('.tif')]
        file_list.sort()

        # # Call the new plotting functions here:
        # self.plot_keypoints_histogram(file_list, folder)
        # self.plot_matches_histogram(file_list, folder, 0.5)
        # self.plot_inliers_outliers(file_list, folder, 0.5, 2500)

        # Read the first image as the initial stitched image
        _, stitched_image, _ = self.read_image(os.path.join(folder, file_list[0]))
        _, _, image = self.read_image(os.path.join(folder, file_list[0]))

        # Iterate through the rest of the images and stitch them with the current stitched image
        for i in range(1, len(file_list)):
            right_gray, right_rgb, right_stitch = self.read_image(os.path.join(folder, file_list[i]))



            # Select the Feature Detector and Descriptor
            if feature_detector == 'SIFT':
                kp_left, des_left = self.SIFT(stitched_image)
                kp_right, des_right = self.SIFT(right_gray)
            elif feature_detector == 'ORB':
                kp_left, des_left = self.ORB(stitched_image)
                kp_right, des_right = self.ORB(right_gray)
            elif feature_detector == 'AKAZE':
                kp_left, des_left = self.AKAZE(stitched_image)
                kp_right, des_right = self.AKAZE(right_gray)
            else:
                raise ValueError("Invalid feature detector specified.")

            matches = self.matcher(kp_left, des_left, stitched_image, kp_right, des_right, right_rgb, 0.6)

            # Compute the Least Squares solution for homography
            # least_squares_H = self.least_squares(matches)
            # least_squares_errors = self.get_error(matches, least_squares_H)


            inliers, H = self.ransac(matches, 0.6, 3000)
            ransac_errors = self.get_error(matches, H)


            # self.plot_comparison(ransac_errors, least_squares_errors, 0.5)



            stitched_image = self.stitch_img(image, right_stitch, H)

            cv2.imshow(f"Stitched Image Set-{i}", stitched_image)
            cv2.waitKey(10)

        cv2.imshow("Final Stitched Image", stitched_image)
        fig = plt.figure()
        plt.imshow(stitched_image)
        plt.show()






def main():
    plt.rcParams['figure.figsize'] = [16, 16]
    stitcher = ImageStitcher()
    stitcher.stitch_images('third', feature_detector ='SIFT')

    cv2.waitKey(0)

if __name__ == '__main__':
    main()