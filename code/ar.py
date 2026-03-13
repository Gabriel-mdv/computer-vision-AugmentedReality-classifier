import numpy as np
import cv2
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from loadVid import loadVid

#Write script for Q3.1

# Load the videos
ar_source = loadVid('../data/ar_source.mov')
book = loadVid('../data/book.mov')

# Load cv_cover as the template for matching
cv_cover = cv2.imread('../data/cv_cover.jpg')

# Get book cover dimensions for cropping aspect ratio
cover_h, cover_w = cv_cover.shape[:2]

# Setup video writer - use book video dimensions and fps
fps = 25
h, w = book[0].shape[:2]
out = cv2.VideoWriter('../results/ar.avi',
					  cv2.VideoWriter_fourcc(*'XVID'),
					  fps, (w, h))

# Process each frame
num_frames = min(len(ar_source), len(book))

for i in range(num_frames):
	book_frame = book[i]
	ar_frame = ar_source[i]

	# Crop ar_frame to match aspect ratio of cv_cover (central region only)
	ar_h, ar_w = ar_frame.shape[:2]
	target_ratio = cover_w / cover_h
	current_ratio = ar_w / ar_h

	if current_ratio > target_ratio:
		# Too wide - crop width
		new_w = int(ar_h * target_ratio)
		start_x = (ar_w - new_w) // 2
		ar_frame_cropped = ar_frame[:, start_x:start_x + new_w]
	else:
		# Too tall - crop height
		new_h = int(ar_w / target_ratio)
		start_y = (ar_h - new_h) // 2
		ar_frame_cropped = ar_frame[start_y:start_y + new_h, :]

	# Resize cropped ar_frame to match cv_cover dimensions
	ar_frame_resized = cv2.resize(ar_frame_cropped, (cover_w, cover_h))

	# Match book frame to cv_cover to find homography
	try:
		matches, locs1, locs2 = matchPics(cv_cover, book_frame)

		# Need at least 4 matches for homography
		if len(matches) < 4:
			out.write(book_frame)
			continue

		# Extract matched point coordinates
		x1 = locs1[matches[:, 0], :]
		x2 = locs2[matches[:, 1], :]

		# Flip from (row, col) to (x, y)
		x1 = x1[:, [1, 0]]
		x2 = x2[:, [1, 0]]

		# Compute homography
		bestH2to1, inliers = computeH_ransac(x1, x2)

		# Composite ar frame onto book frame
		composite_img = compositeH(bestH2to1, ar_frame_resized, book_frame)

		out.write(composite_img)

	except Exception as e:
		# If anything fails for this frame, write original
		print(f"Frame {i} failed: {e}")
		out.write(book_frame)

out.release()
print("ar.avi saved to results/")