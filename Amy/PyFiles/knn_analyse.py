# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def flatten_img(img):
	# note the attempted masking
	# k = img.max()
	# mask = img < (k-2)
	# img = img * mask

	#return img.flatten()
	return img.flatten()
	
def extract_color_histogram(img, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# # handle normalizing the histogram if we are using OpenCV 2.4.X
	# if imutils.is_cv2():
		# hist = cv2.normalize(hist)

	# # otherwise, perform "in place" normalization in OpenCV 3 (I
	# # personally hate the way this is done
	# else:
	cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()
	
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg

	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	
	# use functions defined above
	pixels = flatten_img(image)
	hist = extract_color_histogram(image)

	# append our lists
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits
(X_train, X_test, y_train, y_test) = train_test_split(
	rawImages, labels, test_size= .99, random_state=42)
(Z_train, Z_test, y_train, y_test) = train_test_split(
	features, labels, test_size= .99, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=1,
	n_jobs=-1)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=1,
	n_jobs=-1)
model.fit(Z_train, y_train)
acc = model.score(Z_test, y_test)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
