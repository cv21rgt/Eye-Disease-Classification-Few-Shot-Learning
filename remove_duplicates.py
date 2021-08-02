# import the necessary packages
from imutils import paths
import argparse
import time
import sys
import cv2
import os

# Function that performs "difference hashing"
def dhash(image, hashSize=8):
	# resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(image, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--haystack", required=True,
	help="dataset of images to search through (i.e., the haytack)")
#ap.add_argument("-n", "--needles", required=True,
	#help="set of images we are searching for (i.e., needles)")
args = vars(ap.parse_args())

# grab the paths to the haystack images 
print("[INFO] computing hashes for haystack...")
haystackPaths = list(paths.list_images(args["haystack"]))

# remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
	haystackPaths = [p.replace("\\", "") for p in haystackPaths]
	
# dictionary that will map an image hash value to corresponding image
map_image_to_hash = {}

# loop over the images paths and compute hash values
for p in haystackPaths:
	# load the image from disk
	image = cv2.imread(p)
	# if the image is None then we could not load it from disk (so
	# skip it)
	if image is None:
		continue
	# convert the image to grayscale and compute the hash
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imageHash = dhash(image)
	# update the haystack dictionary
	l = map_image_to_hash.get(imageHash, [])
	l.append(p)
	map_image_to_hash[imageHash] = l

# print the dictionary
#for k, v in map_image_to_hash.items():
    #print(f"{k} ----> {v}")


# dictionary of duplicates
duplicates_dict = {}
for k, v in map_image_to_hash.items():
    if len(map_image_to_hash[k]) > 1:
        duplicates_dict[k] = v

# print dictionary of duplicates
print(duplicates_dict)

# delete the duplicate images
for k, v in duplicates_dict.items():
    for filepath in v[1:]:
        os.remove(filepath)


