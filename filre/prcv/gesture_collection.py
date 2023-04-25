
# Sujai Rajan

# Final Project: Gesture Recognition System

# CS5330 - Pattern Recognition and Computer Vision


# import statements
import cv2
import numpy as np
import os


# Create Folder to store the images if it does not exist
if not os.path.exists('./collected_images'):
    os.mkdir('./collected_images')


# initialize the camera
cap = cv2.VideoCapture(0)  # set camera to desired index


# set the camera resolution to 1000x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


# initialize the weight for running average 
aweight = 0.5   # alpha weight
num_frames = 0  # number of frames
bg = None       # background model


# Helps the algorithm to adapt the changes in the background and lighting conditions
# Function to calculate the running average over the background 
def run_avg(img,aweight):
    # initialize the global background
    global bg
    # check if background is None
    if bg is None:
        # initialize the background as a copy of the image passed in 
        bg = img.copy().astype('float')
        return
    # compute the weighted average, accumulate it and update the background
    cv2.accumulateWeighted(img,bg,aweight)


# Uses the background model to segment the hand from the image (largest contour is considered as the hand)
# Function to segment the region of hand in the image
def segment(img,thres=25):
    # initialize the global background
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype('uint8'),img)
    # threshold the diff image so that we get the foreground
    _, thresholded = cv2.threshold(diff,thres,255,cv2.THRESH_BINARY)
    # get the contours in the thresholded image
    contours,_ = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # return None, if no contours detected
    if len(contours) == 0:
        return
    # based on contour area, get the maximum contour which is the hand
    else:
        segmented = max(contours,key = cv2.contourArea)
    # return the thresholded image and segmented region of the hand
    return (thresholded,segmented)

# Initialize flag and variables
flag = 0
i = 0
N1 = 0
min_area = 5000

# Main Function Loop
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Main Logic when camera is on
    if ret ==True:

        # Flip the frame
        frame = cv2.flip(frame, 1)
        # Clone the frame to draw on it
        clone = frame.copy()
        # Get the height and width of the frame
        (height, width) = frame.shape[:2]

        # Get the Region of Interest
        roi = frame[100:300, 300:500]   # Need to change according to the camera position and resolution
        
        # Convert the roi to grayscale and apply Gaussian blur to smoothen it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # For the first 100 frames, we will calculate the average of the background. (Keep the ROI frame clear)
        if num_frames < 100:
            cv2.putText(clone, "Calculating background model...", (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
            # print(num_frames)
            run_avg(gray, aweight)

            if num_frames == 99:
                print("Background Calculated")
                print("Place your hand in the box to collect images")


        # Once the background is calculated, Segment the hand region
        else:
            # segment the hand region 
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and segmented region
                (thresholded, segmented) = hand

                # draw the contours of the segmented hand on the cloned frame
                cv2.drawContours(clone, [segmented + (300, 100)], -1, (0, 0, 255))

                # Display the thresholded image of the hand
                cv2.imshow("Thesholded Frame", thresholded)

                # Find the contours in the thresholded image of the hand and draw them on the cloned frame
                contours, _= cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(clone,contours,-1,(255, 0, 0),4)

                # Find the largest contour and draw it on the cloned frame
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # print(area)

                    # If the area of the contour is greater than minimum area, then it is the hand
                    if area > min_area:
                        # put text on the frame above roi to indicate that hand is detected
                        cv2.putText(clone, "Hand Detected", (300, 350), cv2.FONT_HERSHEY_DUPLEX  , 1,  (0, 165, 255), 2)
                        
                        # Draw the bounding rectangle around the hand 
                        x,y,w,h = cv2.boundingRect(cnt)
                        cv2.rectangle(clone,(x,y),(x+w,y+h),(128, 0, 128),4)
                        
                        # Resize the image to 64x64 and save it
                        to_save = cv2.resize(thresholded,(64,64))
                        cv2.imwrite('./collected_images/'+str(i)+'.jpg',to_save)
                        i = i + 1

        # Draw the ROI on the cloned frame
        cv2.rectangle(clone, (300, 100), (500, 300), (0, 255, 0), 2)
        cv2.putText(clone, "Gesture Recognition-Image Collection", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # Display the resulting frame 
        cv2.imshow('Camera Input', clone)

        # Press 'q' to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
    
    # Break the loop when camera is off
    else:
        print("Camera is off")
        break

# Release the camera
cap.release()

# Destroy all windows
cv2.destroyAllWindows()


