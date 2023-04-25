# Sujai Rajan

# Final Project: Gesture Recognition System

# CS5330 - Pattern Recognition and Computer Vision


# import statements
from gesture_prediction_calc import *
import cv2
import warnings


# Ignore the warnings
warnings.filterwarnings('ignore') 


# initialize the camera
cap = cv2.VideoCapture(0)  # set camera to desired index


# set the camera resolution to 1000x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


# List of labels defined for gesture recognition
label = ['0','1','+','-','*','/','Confirm','**','%','Clear','2','3','4','5','6','7','8','9']


# Loading the trained model for gesture recognition
model = load_model('./gesture_recognition_model.h5')


# initialize the weight for running average 
aweight = 0.5   # alpha weight
num_frames = 0  # number of frames
bg = None       # background model

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

# Initialize the variables
count = 0
min_area = 5000
first_number = ""
operator = ""
second_number = ""


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
        # Create a blank extra frame to draw contours
        clone_extra = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
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
                count += 1

                # draw the contours of the segmented hand on the cloned frame
                cv2.drawContours(clone, [segmented + (300, 100)], -1, (0, 0, 255))

                # Display the thresholded image of the hand
                cv2.imshow("Thesholded Frame", thresholded)

                # Find the contours in the thresholded image of the hand 
                contours, _= cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                
                # # Draw the contours on the blank screen
                # cv2.drawContours(clone_extra,contours,-1,(0, 100, 0),4)
                # cv2.imshow("Detected Contours", clone_extra)

                print(count)

                # Display 'Calculator Ready' for 3 Secs
                if count < 90:
                    cv2.putText(clone, 'Calculator Ready', (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)

                # Display 'Enter the first Number' for 2 Secs
                elif count > 90 and count < 150:
                    cv2.putText(clone, 'Enter the first Number', (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)

                # Display 'Confirmed' for 2 Secs and collect the first number
                elif count > 481 and count < 540:
                    cv2.putText(clone, "Confirmed", (50, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
                    wor = "The first number is " + first_number
                    cv2.putText(clone, wor, (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)

                # Display 'Enter the operator' for 4 Secs
                elif count > 540 and count < 660:
                    cv2.putText(clone, "Enter the operator", (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)

                # Display 'Confirmed' for 2 Secs and collect the operator
                elif count > 721 and count < 781:
                    cv2.putText(clone, "Confirmed", (50, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
                    cv2.putText(clone, operator, (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)

                # Display 'Enter the Second Number' for 2 Secs
                elif count > 781 and count < 840:
                    cv2.putText(clone, 'Enter the Second Number', (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)

                # Display 'Confirmed' for 2 Secs and collect the second number
                elif count > 1201 and count < 1261:
                    cv2.putText(clone, "Confirmed", (50, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
                    wor = "The second number is " + second_number
                    cv2.putText(clone, wor, (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)

                # Display the result till program ends 
                elif count > 1300:
                    res = get_result(first_number,operator,second_number)
                    in_line = first_number + operator + second_number + " = " + str(res)
                    cv2.putText(clone, "The answer is ", (50, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
                    cv2.putText(clone,in_line,(50,400),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
                
                # # Clear the screen after 20 seconds
                # elif count > 1900:
                #     cap.release()
                #     cv2.destroyAllWindows()


                # Find the largest contour and draw it on the cloned frame
                for cnt in contours:
                    if cv2.contourArea(cnt) > min_area:
                        cv2.putText(clone, "Hand Detected", (300, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)

                        # Inputing the first number for 12 Secs and 2 Secs for each character
                        if count > 150 and count < 481:
                            # Display the first number on the screen
                            cv2.putText(clone, 'Enter the first Number', (50, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
                            cv2.putText(clone, first_number, (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (154, 205, 50), 2)

                            # Loop every 2 Secs
                            if count % 60 == 0:
                                # Get the prediction from the model
                                pred = get_prediction(thresholded)

                                # If the prediction is not Confirm or Clear, add the prediction to the first number
                                if pred != "Confirm" and pred != "Clear":
                                    first_number = first_number + pred

                                # If the prediction is Clear, clear the first number and send loop back to 'Enter the first number'
                                elif pred == "Clear":
                                    count = 91
                                    first_number = ""
                                
                                # If first number is defined, continue to the next step
                                else:
                                    count = 481

                        # Inputing the operator for 2 Secs
                        elif count > 660 and count < 721:
                            cv2.putText(clone, 'Enter the operator', (50, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
                            
                            # Get the prediction from the model and set it as the operator and display it on the screen
                            pred = get_prediction(thresholded)
                            operator = pred
                            cv2.putText(clone, operator, (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (154, 205, 50), 2)

                            # If the prediction is Clear, clear the operator and send loop back to 'Enter the operator'
                            if pred == "Clear":
                                count = 661
                                operator = ""
                            
                            # If operator is defined, continue to the next step
                            else:
                                count = 721


                        # Inputing the second number for 12 Secs and 2 Secs for each character
                        elif count > 841 and count <1201:
                            # Display the second number on the screen
                            cv2.putText(clone, 'Enter the second Number', (50, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
                            cv2.putText(clone, second_number, (50, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (154, 205, 50), 2)

                            # Loop every 2 Secs
                            if count % 60 == 0:
                                # Get the prediction from the model
                                pred = get_prediction(thresholded)

                                # If the prediction is not Confirm or Clear, add the prediction to the second number
                                if pred != "Confirm" and pred != "Clear":
                                    second_number = second_number + pred

                                # If the prediction is Clear, clear the second number and send loop back to 'Enter the second number'    
                                elif pred == "Clear":
                                    count = 782
                                    second_number = ""

                                 # If second number is defined, continue to the next step    
                                else:
                                    count = 1201

                        # Wait to see if user wants to Clear and start over
                        elif count > 1300:

                            # Loop every 2 Secs
                            if count % 60 == 0:
                                # Get the prediction from the model
                                pred = get_prediction(thresholded)

                                # If the prediction is Clear, clear the variables and send loop back to the start
                                if pred == "Clear":
                                    count = 0
                                    first_number = ""
                                    operator = ""
                                    second_number = ""


        # Draw the ROI on the cloned frame
        cv2.rectangle(clone, (300, 100), (500, 300), (0, 255, 0), 2)
        cv2.putText(clone, "Gesture Recognition - Calculator", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

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

# Print the operator and the numbers with the result
print(first_number, operator, second_number,"=",get_result(first_number,operator,second_number))
# cv2.waitKey()

# Destroy all windows
cv2.destroyAllWindows()