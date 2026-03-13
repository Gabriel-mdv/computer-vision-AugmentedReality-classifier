import numpy as np
import cv2
#Import necessary functions only

def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of points
	N = x1.shape[0]
	A = []
 
	for i in range(N):
		x1_i = x1[i]
		x2_i = x2[i]
		A.append([-x1_i[0], -x1_i[1], -1, 0, 0, 0, x2_i[0]*x1_i[0], x2_i[0]*x1_i[1], x2_i[0]])
		A.append([0, 0, 0, -x1_i[0], -x1_i[1], -1, x2_i[1]*x1_i[0], x2_i[1]*x1_i[1], x2_i[1]])
  
	A = np.array(A)
 
	#Compute SVD of A
	U, S, Vt = np.linalg.svd(A)
	H = Vt[-1].reshape(3, 3)
	H2to1 = H / H[2, 2]


	return H2to1

def computeH_norm(x1, x2):
	#Q3.7
	#Compute the centroid of the points
	mean1 = np.mean(x1, axis=0)
	mean2 = np.mean(x2, axis=0)

	#Shift the origin of the points to the centroid
	x1_shifted = x1 - mean1
	x2_shifted = x2 - mean2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	scale1 = np.sqrt(2) / np.max(np.sqrt(np.sum(x1_shifted**2, axis=1)))
	scale2 = np.sqrt(2) / np.max(np.sqrt(np.sum(x2_shifted**2, axis=1)))

	#Similarity transform 1
	T1 = np.array([[scale1, 0, -scale1*mean1[0]],
				   [0, scale1, -scale1*mean1[1]],
				   [0, 0, 1]])

	#Similarity transform 2
	T2 = np.array([[scale2, 0, -scale2*mean2[0]],
				   [0, scale2, -scale2*mean2[1]],
				   [0, 0, 1]])

	#Compute homography
	x1_norm = (x1_shifted) * scale1
	x2_norm = (x2_shifted) * scale2
	H_norm = computeH(x1_norm, x2_norm)

	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H_norm @ T2

	return H2to1

def computeH_ransac(x1, x2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	N = x1.shape[0]
	max_iters = 1000
	threshold = 4.0
	best_inliers = np.zeros(N)
	bestH2to1 = None

	for i in range(max_iters):
		#Randomly sample 4 point pairs
		idx = np.random.choice(N, 4, replace=False)
		x1_sample = x1[idx]
		x2_sample = x2[idx]

		#Compute homography from sample
		H = computeH_norm(x1_sample, x2_sample)

		#Convert x2 to homogeneous coordinates and project
		x2_hom = np.column_stack((x2, np.ones(N)))
		x1_proj = (H @ x2_hom.T).T

		#Convert back to inhomogeneous coordinates
		x1_proj = x1_proj[:, :2] / x1_proj[:, 2:3]

		#Compute distances between projected and actual points
		distances = np.sqrt(np.sum((x1_proj - x1)**2, axis=1))

		#Find inliers
		inliers = (distances < threshold).astype(int)

		#Update best homography if more inliers found
		if np.sum(inliers) > np.sum(best_inliers):
			best_inliers = inliers
			bestH2to1 = H

	#Recompute homography using all inliers
	inlier_idx = best_inliers == 1
	bestH2to1 = computeH_norm(x1[inlier_idx], x2[inlier_idx])
	inliers = best_inliers

	return bestH2to1, inliers

def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	H_inv = np.linalg.inv(H2to1)

	#Create mask of same size as template
	mask = np.ones_like(template, dtype=np.uint8) * 255

	#Warp mask by appropriate homography
	h, w = img.shape[:2]
	warped_mask = cv2.warpPerspective(mask, H_inv, (w, h))

	#Warp template by appropriate homography
	warped_template = cv2.warpPerspective(template, H_inv, (w, h))

	#Use mask to combine the warped template and the image
	composite_img = img.copy()
	composite_img[warped_mask > 0] = warped_template[warped_mask > 0]
	
	return composite_img
