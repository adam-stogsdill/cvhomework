import os
import cv2
import numpy as np

from sklearn.metrics import confusion_matrix

IMAGE_DIRECTORY = ""
TRAINING_IMAGE_NAME = "mural01.jpg"
QUERY_IMAGE_NAME = "mural02.jpg"

#points = np.asarray([[294.0, 243.0, 1.0], [462.0, 232.0, 1.0]])

actual = []
predicted = []

def perform(TRAINING_IMAGE_NAME, QUERY_IMAGE_NAME, points, offset):
    #file_path = os.path.join(IMAGE_DIRECTORY, TRAINING_IMAGE_NAME)
    #assert (os.path.exists(file_path))
    bgr_train = cv2.imread(TRAINING_IMAGE_NAME)  # Get training image
    bgr_train_vis = bgr_train.copy()
    '''for x in range(4):
        cv2.circle(bgr_train_vis, (int(points[x,0]), int(points[x,1])), 3, (255,0,0), 5)'''
    #file_path = os.path.join(IMAGE_DIRECTORY, QUERY_IMAGE_NAME)
    #assert (os.path.exists(file_path))
    bgr_query = cv2.imread(QUERY_IMAGE_NAME)  # Get query image

    actual.append(1 if "negative" in QUERY_IMAGE_NAME else 0)

    # Show input images.
    #cv2.imshow("Training image", bgr_train_vis)
   # cv2.imwrite("TrainingImageWithAnnotatinons.png", bgr_train_vis)
    #cv2.imshow("Query image", bgr_query)

    # Extract keypoints and descriptors.

    orb = cv2.ORB_create(nfeatures=2000)

    kp_train, desc_train = detect_features(bgr_train)
    kp_query, desc_query = detect_features(bgr_query)

    matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)

    # Match query image descriptors to the training image.
    # Use k nearest neighbor matching and apply ratio test.
    matches = matcher.knnMatch(desc_query, desc_train, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    matches = good
    print("Number of raw matches between training and query: ", len(matches))

    bgr_matches = cv2.drawMatches(
        img1=bgr_query, keypoints1=kp_query,
        img2=bgr_train, keypoints2=kp_train,
        matches1to2=matches, matchesMask=None, outImg=None)
    #cv2.imshow("All matches", bgr_matches)

    # show_votes(bgr_query, kp_query, bgr_train, kp_train, matches)

    matches = find_cluster(bgr_query, kp_query, bgr_train, kp_train, matches,
                           show_votes=True)
    print("Number of matches in the largest cluster:", len(matches))
    print("Matches:",matches)
    print("keypoints:", kp_query)

    # Draw matches between query image and training image.
    bgr_matches = cv2.drawMatches(
        img1=bgr_query, keypoints1=kp_query,
        img2=bgr_train, keypoints2=kp_train,
        matches1to2=matches, matchesMask=None, outImg=None)
    #cv2.imshow("Matches in largest cluster", bgr_matches)

    # Calculate an affine transformation from the training image to the query image.
    A_train_query, inliers, dst_points, src_points = calc_affine_transformation(matches, kp_train, kp_query)


    H, _ = cv2.findHomography(src_points, dst_points)

    first_mural_warped = cv2.warpPerspective(bgr_query, H, (1000, 500))
    #cv2.imshow("ah",first_mural_warped)

    print(H)


    # Apply the affine warp to warp the training image to the query image.
    if A_train_query is not None and len(inliers) > 3:
        # Object detected! Warp the training image to the query image and blend the images.
        print("Object detected! Found %d inlier matches" % sum(inliers))
        warped_training = cv2.warpAffine(
            src=bgr_train, M=A_train_query,
            dsize=(bgr_query.shape[1], bgr_query.shape[0]))

        print(A_train_query.shape)
        print(A_train_query @ points.T)

        p = A_train_query @ points.T

        print("P:\n", p)

        for x in range(4):
            print(p[0,x])
            cv2.circle(bgr_query, (int(p[0,x]), int(p[1,x])), 3, (255,0,0), 5)

        cv2.imwrite("found_object_in_" + QUERY_IMAGE_NAME, bgr_query)

        # Blend the images.
        blended_image = bgr_query / 2
        blended_image[:, :, 1] += warped_training[:, :, 1] / 2
        blended_image[:, :, 2] += warped_training[:, :, 2] / 2
        #cv2.imshow("Blended", bgr_query)
        #cv2.imshow("matches", bgr_matches)
        #cv2.imshow("warped training", warped_training)
        predicted.append(0)
    else:
        print("Object not detected; can't fit an affine transform")
        predicted.append(1)
    #cv2.waitKey(0)
    return H
    


# Detect features in the image and return the keypoints and descriptors.
def detect_features(bgr_img, show_features=False):
    orb = cv2.ORB_create()

    # Extract keypoints and descriptors from image.
    gray_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_image, mask=None)

    # Optionally draw detected keypoints.
    if show_features:
        # Possible flags: DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, DRAW_MATCHES_FLAGS_DEFAULT
        bgr_display = bgr_img.copy()
        cv2.drawKeypoints(image=bgr_display, keypoints=keypoints,
                          outImage=bgr_display,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("Features", bgr_display)
        print("Number of keypoints: ", len(keypoints))
        cv2.waitKey(0)

    return keypoints, descriptors


# Given the proposed matches, each match votes into a quantized "pose" space. Find the
# bin with the largest number of votes, and return the matches within that bin.
def find_cluster(query_img, keypoints_query, train_img, keypoints_train, matches,
                 show_votes=False):
    hq = query_img.shape[0]
    wq = query_img.shape[1]

    max_scale = 4.0  # Scale differences go from 0 to max_scale

    # Our accumulator array is a 4D array of empty lists. These are the number of bins
    # for each of the dimensions.
    num_bins_height = 5
    num_bins_width = 5
    num_bins_scale = 5
    num_bins_ang = 8

    # It is easier to have a 1 dimensional array instead of a 4 dimensional array.
    # Just convert subscripts (h,w,s,a) to indices idx.
    size_acc = num_bins_height * num_bins_width * num_bins_scale * num_bins_ang
    acc_array = [[] for idx in range(size_acc)]

    ht = train_img.shape[0]
    wt = train_img.shape[1]

    # Vote into accumulator array.
    for match in matches:
        qi = match.queryIdx  # Index of query keypoint
        ti = match.trainIdx  # Index of training keypoint that matched

        # Get data for training image.
        kp_train = keypoints_train[ti]
        at = kp_train.angle
        st = kp_train.size
        pt = np.array(kp_train.pt)  # training keypoint location
        mt = np.array([wt / 2, ht / 2])  # Center of training image
        vt = mt - pt  # Vector from keypoint to center

        # Get data for query image.
        kp_query = keypoints_query[qi]
        aq = kp_query.angle
        sq = kp_query.size
        pq = np.array(kp_query.pt)

        # Rotate and scale the vector to the marker point.
        scale_factor = sq / st
        angle_diff = aq - at
        angle_diff = (angle_diff + 360) % 360  # Force angle to between 0..360 degrees
        vq = rotate_and_scale(vt, scale_factor, angle_diff)
        mq = pq + vq

        if show_votes:
            print("Scale diff %f, angle diff %f" % (scale_factor, angle_diff))

            # Display training image.
            train_img_display = train_img.copy()
            cv2.drawKeypoints(image=train_img_display, keypoints=[kp_train],
                              outImage=train_img_display,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawMarker(img=train_img_display, position=(int(mt[0]), int(mt[1])),
                           color=(255, 0, 0),
                           markerType=cv2.MARKER_DIAMOND)
            cv2.line(img=train_img_display,
                     pt1=(int(pt[0]), int(pt[1])), pt2=(int(mt[0]), int(mt[1])),
                     color=(255, 0, 0), thickness=2)
            #cv2.imshow("Training keypoint", train_img_display)

            # Display query image.
            query_img_display = query_img.copy()
            cv2.drawKeypoints(image=query_img_display, keypoints=[kp_query],
                              outImage=query_img_display,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.line(img=query_img_display,
                     pt1=(int(pq[0]), int(pq[1])), pt2=(int(mq[0]), int(mq[1])),
                     color=(255, 0, 0), thickness=2)
            #cv2.imshow("Query keypoint", query_img_display)
            cv2.waitKey(100)

        # Compute the cell of the accumulator array, that this match should be stored in.
        row_subscript = int(round(num_bins_height * (mq[1] / hq)))
        col_subscript = int(round(num_bins_width * (mq[0] / wq)))
        if row_subscript >= 0 and row_subscript < num_bins_height and col_subscript >= 0 and col_subscript < num_bins_width:
            scale_subscript = int(num_bins_scale * (scale_factor / max_scale))
            if scale_subscript > num_bins_scale:
                scale_subscript = num_bins_scale - 1

            ang_subscript = int(num_bins_ang * (angle_diff / 360))
            # print(row_subscript,col_subscript, scale_subscript, ang_subscript)

            # Note: the numpy functions ravel_multi_index(), and unravel_index() convert
            # subscripts to indices, and vice versa.
            idx = np.ravel_multi_index(
                (row_subscript, col_subscript, scale_subscript, ang_subscript),
                (num_bins_height, num_bins_width, num_bins_scale, num_bins_ang))

            acc_array[idx].append(match)

    # Count matches in each bin.
    counts = [len(acc_array[idx]) for idx in range(size_acc)]

    # Find the bin with maximum number of counts.
    idx_max = np.argmax(np.array(counts))

    # Return the matches in the largest bin.
    return acc_array[idx_max]


# Calculate an affine transformation from the training image to the query image.
def calc_affine_transformation(matches_in_cluster, kp_train, kp_query):
    if len(matches_in_cluster) < 3:
        # Not enough matches to calculate affine transformation.
        return None, None

    # Estimate affine transformation from training to query image points.
    # Use the "least median of squares" method for robustness. It also detects outliers.
    # Outliers are those points that have a large error relative to the median of errors.
    src_pts = np.float32([kp_train[m.trainIdx].pt for m in matches_in_cluster]).reshape(
        -1, 1, 2)
    dst_pts = np.float32([kp_query[m.queryIdx].pt for m in matches_in_cluster]).reshape(
        -1, 1, 2)
    A_train_query, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.LMEDS)

    return A_train_query, inliers, src_pts, dst_pts


def rotate_and_scale(vt, scale_factor, angle_diff):
    theta = np.radians(angle_diff)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    vq = R @ vt
    vq = vq * scale_factor
    return vq


if __name__ == "__main__":
    '''import os
    for f in os.listdir("./query_images/"):
        if "printer" in f:
            continue
        QUERY_IMAGE_NAME = f
        main()
    print(confusion_matrix(actual, predicted))'''
    perform("mural02.jpg", "mural03.jpg", np.asarray([[521, 182, 1], [716, 184, 1], [705, 400, 1], [477, 375, 1]]), None)
