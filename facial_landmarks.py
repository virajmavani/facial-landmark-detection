from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while(cap.isOpened()):
	ret, image = cap.read()
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
 
	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	if cv2.waitKey(1) & 0xFF == ord ('q'):
		break

cap.release()
cv2.destroyAllWindows()