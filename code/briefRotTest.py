import numpy as np
import cv2
import scipy
from helper import plotMatches
from matchPics import matchPics
import matplotlib.pyplot as plt


#Q2.4
#Read the image and convert to grayscale, if necessary, you can use OpenCV
img = cv2.imread('../data/cv_cover.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# store the number of matches for each ange in a histogram
angles = []
match_counts = []

for i in range(36):
	#Rotate Image
	angle = i*10
	rotated_img = scipy.ndimage.rotate(img, angle)
 
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, rotated_img)

	#Update histogram
	angles.append(angle)
	match_counts.append(len(matches))
 
	# save visualization at three different orientations (0, 90, 180)
	if angle in [0, 90, 180]:
      		plotMatches(img, rotated_img, matches, locs1, locs2)
	
 
#Display histogram
plt.bar(angles, match_counts)
plt.xlabel('Rotation Angle')
plt.ylabel('Number of Matches')
plt.title('Brief Descriptor Matching Results')
plt.show()

