import cv2
import numpy as np
from orb import perform

def fuse_color_images(A, B):
    assert(A.ndim == 3 and B.ndim == 3)
    assert(A.shape == B.shape)
    # Allocate result image.
    C = np.zeros(A.shape, dtype=np.uint8)
    # Create masks for pixels that are not zero.
    A_mask = np.sum(A, axis=2) > 0
    B_mask = np.sum(B, axis=2) > 0
    # Compute regions of overlap.
    A_only = A_mask & ~B_mask
    B_only = B_mask & ~A_mask
    A_and_B = A_mask & B_mask
    C[A_only] = A[A_only]
    C[B_only] = B[B_only]
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]
    return 


# Get first image
icurrent = cv2.imread("mural01.jpg")

prev_image_height = icurrent.shape[0]
prev_image_width = icurrent.shape[1]

image_corners = np.asarray([[153, 120], [382, 69], [306, 504], [24, 479]])

desired_width = (1/0.005) * 1.593
desired_height = (1/0.005) * 2.178

print(desired_width, desired_height)

desired_corners = np.asarray([[0,0], [desired_width, 0], [desired_width, desired_height], [0, desired_height]])

desired_corners[:,0] += 50       # Add x offset
desired_corners[:,1] += 100      # Add y offset

H_current_mosaic, _ = cv2.findHomography(image_corners, desired_corners)

output_width = int(desired_width * 5)
output_height = int(desired_height * 2)

first_mural_warped = cv2.warpPerspective(icurrent, H_current_mosaic, (output_width, output_height))

iprev = icurrent
H_prev_mosaic = H_current_mosaic

mural_0_1_points = np.asarray([[521, 182, 1], [716, 184, 1], [705, 400, 1], [477, 375, 1]])


# Loop through rest of images
for i in range(2, 13):
    if i < 10:
        icurrent = cv2.imread("mural0"+str(i)+".jpg")
    else:
        icurrent = cv2.imread("mural"+str(i)+".jpg")

    H_current_mosaic = H_prev_mosaic @ perform("mural01.jpg", "mural0"+str(i)+".jpg", points=mural_0_1_points, prev_image_width / 2)

    current_mural_warped = cv2.warpPerspective(icurrent, H_current_mosaic, (output_width, output_height))
    cv2.imshow("ah3", first_mural_warped)
    cv2.imshow("ah2", current_mural_warped)
    cv2.waitKey(0)


    
    

#cv2.imshow("ah", bgr_ortho)
#cv2.waitKey(0)