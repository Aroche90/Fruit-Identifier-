# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# def flatten_image(img):
	# return cv2(img).flatten()
	
# def grayscale(img):
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# return cv2.imshow("gray", gray)
	
#gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# grab the list of images that we'll be describing
print("[INFO] describing images...")
image_paths = list(paths.list_images(args["dataset"]))

features = []
rawmages = []
labels = []

for image_path in image_paths:
	label = image_path.split(os.path.sep)[-1].split(".")[0]
	labels.append(label)
	image = cv2.imread(image_path)
    chans = cv2.split(image)
	colors = ("b", "g", "r")
	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and
		# concatenate the resulting histograms for each
		# channel
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		features.extend(hist)
		

# def extract_color_histogram(img):
	# # extract a 3D color histogram from the HSV color space using
	# # the supplied number of `bins` per channel
	# hsv = cv2.cvtColor(image_fun, cv2.COLOR_BGR2HSV)
	# hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		# [0, 180, 0, 256, 0, 256])
	# cv2.normalize(hist, hist)

	# # return the flattened histogram as the feature vector
	# return hist.flatten()
