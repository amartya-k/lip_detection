from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
 
	# if the colors list is None, initialize it with a unique
	# colors
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
            (168, 100, 168), (158, 163, 32),
            (163, 38, 32), (180, 42, 220)]
    
    #coordiantes of lips
    pts=shape[48:68]
    hull = cv2.convexHull(pts)

    #choose any colour 
    #draw the contours
    cv2.drawContours(overlay, [hull], -1, colors[5], -1)
    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
 
	# return the output image
    return output        

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect faces in the image
rects = detector(gray, 1)

#iterate through faces

for (i, rect) in enumerate(rects):
	#extract the coordinates of facical parts
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	#clone = image.copy()
	#cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)

	#magic happens here
	output = visualize_facial_landmarks(image, shape)
	#cv2.imshow("Image", clone)
	cv2.imshow("Image", output)
	cv2.waitKey(0)