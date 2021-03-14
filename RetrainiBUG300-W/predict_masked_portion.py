# import the necessary packages
# modified predict_eyes.py from https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/
# also used sample .py from dlib for face_landmark_detection  https://github.com/davisking/dlib/blob/master/python_examples/face_landmark_detection.py
import argparse
import imutils
import dlib
import glob
import sys
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--input-image", required=True,
	help="path to input image to annotate with landmarks")
ap.add_argument("-o", "--output-image", required=True,
	help="path to output image of annotated image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

win = dlib.image_window()

img = dlib.load_rgb_image(args["input_image"])

win.clear_overlay()
win.set_image(img)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for k, d in enumerate(dets):
 print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
	 k, d.left(), d.top(), d.right(), d.bottom()))
 # Get the landmarks/parts for the face in box d.
 shape = predictor(img, d)
 print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
										   shape.part(1)))
 # Draw the face landmarks on the screen.
 win.add_overlay(shape)

win.add_overlay(dets)
dlib.hit_enter_to_continue()




