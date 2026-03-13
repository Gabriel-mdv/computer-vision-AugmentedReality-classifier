import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
#Complete functions above this line before this step
def matchPics(I1, I2):
	#I1, I2 : Images to match
	

	#Convert Images to GrayScale
	
	I1 = skimage.color.rgb2gray(I1)
	I2 = skimage.color.rgb2gray(I2)
	
	#Detect Features in Both Images
	loc1 = corner_detection(I1)
	loc2 = corner_detection(I2)
	
	#Obtain descriptors for the computed feature locations
	
	desc1, locs1 = computeBrief(I1, loc1)
	desc2, locs2 = computeBrief(I2, loc2)

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2)

	return matches, locs1, locs2
