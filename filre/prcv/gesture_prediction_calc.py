
# Sujai Rajan

# Final Project: Gesture Recognition System

# CS5330 - Pattern Recognition and Computer Vision


# import statements
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import img_to_array


# List of labels defined for gesture recognition
label = ['0','1','+','-','*','/','Confirm','**','%','Clear','2','3','4','5','6','7','8','9']


# Loading the trained model for gesture recognition
model = load_model('./gesture_recognition_model.h5')


# Function to find the prediction of the gesture and return the label
def get_prediction(img):
    # resize the image to 64x64
    for_pred = cv2.resize(img,(64,64))
    # convert the image to array
    x = img_to_array(for_pred)
    # normalize the image
    x = x/255.0
    # reshape the image to 4D tensor (1,64,64,3)
    x = x.reshape((1,) + x.shape)
    # find the prediction of the gesture using the trained model
    pred = str(label[np.argmax(model.predict(x))])
    # return the prediction label
    return pred


# Calculator function to perform the arithmetic operations
def get_result(num_1,operator,num_2):

	# Check if numbers have only integers
	if num_1.isdigit() == False or num_2.isdigit() == False:
		print("Not a valid number")
		return -1
	else:
		# convert the numbers to integer
		a = int(num_1)
		b = int(num_2)

		# Define the arithmetic operations
		if operator == "*":
			res = a * b
		elif operator == "/":
			res = a/b
		elif operator == "+":
			res = a + b
		elif operator == "%":
			res = a % b
		elif operator == "-":
			res = a - b
		elif operator == "**":
			res = a**b
		else:
			print("Not a valid operator")
		
		# return the result
		return res
