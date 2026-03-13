import numpy as np
import cv2
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q3.9


# Read images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# Compute homography automatically using matchPics and computeH_ransac
matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# Extract matched point coordinates
x1 = locs1[matches[:, 0], :]
x2 = locs2[matches[:, 1], :]

# Note: locs are in (row, col) = (y, x), flip to (x, y) for homography
x1 = x1[:, [1, 0]]
x2 = x2[:, [1, 0]]

bestH2to1, inliers = computeH_ransac(x1, x2)

# Resize hp_cover to match cv_cover dimensions so it fills the book correctly
hp_cover_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

# Composite the warped image onto the desk
composite_img = compositeH(bestH2to1, hp_cover_resized, cv_desk)

# Display result
cv2.imshow('HarryPotterized', composite_img)
cv2.waitKey(0)
cv2.destroyAllWindows()