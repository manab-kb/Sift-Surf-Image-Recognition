import cv2
import matplotlib.pyplot as plt
import numpy as np

def surf_detector(test_image):
    image1 = cv2.imread('./Images/Face_Testing.jpg')
    # To recognize multiple images, create a list of images here
    
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Convert the image to gray scale
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    # Add Scale Invariance and Rotational Invariance to test image
    test_image = cv2.pyrDown(test_image)
    num_rows, num_cols = test_image.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
    test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

    surf = cv2.xfeatures2d.SURF_create(800)

    train_keypoints, train_descriptor = surf.detectAndCompute(image1, None)
    test_keypoints, test_descriptor = surf.detectAndCompute(test_gray, None)

    keypoints_without_size = np.copy(image1)
    keypoints_with_size = np.copy(image1)

    cv2.drawKeypoints(image1, train_keypoints, keypoints_without_size, color = (0, 255, 0))
    cv2.drawKeypoints(image1, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

    # Perform the matching between the SURF descriptors of the training image and the test image
    matches = bf.match(train_descriptor, test_descriptor)

    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key = lambda x : x.distance)

    result = cv2.drawMatches(image1, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)

    # Print total number of matching points between the training and query images
    if (len(matches) > 100):
        print("Recognized")
    else:
        print("Not Recognized")
