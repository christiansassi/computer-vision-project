import cv2
import numpy as np  
import math
from typing import Union
import inspect
import logging

def filter_matches(matches: list[list], left_frame_keypoints: tuple[cv2.KeyPoint], right_frame_keypoints: tuple[cv2.KeyPoint], value: float, angle: float) -> list[list]:

    # Applying ratio test and filtering out the good matches
    matches = [[match1] for match1, match2 in matches if match1.distance < value * match2.distance]

    # Filter matches based on angles
    _matches = []

    for index, match in enumerate(matches):
        fp_x, fp_y = left_frame_keypoints[match[0].queryIdx].pt
        sp_x, sp_y = right_frame_keypoints[match[0].trainIdx].pt

        inclination = math.degrees(math.atan((sp_y-fp_y)/(sp_x-fp_x)))

        if abs(inclination) > angle: 
            logging.debug(f"Match {index} with inclination: {inclination} and distance {match[0].distance} [REMOVED]\n")
            continue

        _matches.append(match)
        logging.debug(f"Match {index} with inclination: {inclination} and distance {match[0].distance}\n")
    
    matches = _matches

    return matches

def find_matches(left_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, right_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, k: int) -> tuple[list[list], tuple[cv2.KeyPoint], tuple[cv2.KeyPoint]]:

    # Copy the image
    _left_frame = left_frame.copy()
    # Convert to gray
    _left_frame = cv2.cvtColor(src=_left_frame, code=cv2.COLOR_BGR2GRAY)

    # Copy the image
    _right_frame = right_frame.copy()
    # Convert to gray
    _right_frame = cv2.cvtColor(src=_right_frame, code=cv2.COLOR_BGR2GRAY)

    # Using SIFT to find the keypoints and decriptors in the images
    sift = cv2.SIFT_create()
    left_frame_keypoints, left_frame_descriptors = sift.detectAndCompute(_left_frame, None)
    right_frame_keypoints, right_frame_descriptors = sift.detectAndCompute(_right_frame, None)

    # Using Brute Force matcher to find matches
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(queryDescriptors=left_frame_descriptors, trainDescriptors=right_frame_descriptors, k=k)

    return matches, left_frame_keypoints, right_frame_keypoints

def find_homography(matches: list[list], left_frame_keypoints: tuple[cv2.KeyPoint], right_frame_keypoints: tuple[cv2.KeyPoint], ransacReprojThreshold: float) -> np.ndarray:

    # Check if matches are more than 4
    assert len(matches) >= 4, "Not enough matches"

    # Storing coordinates of points corresponding to the matches found in both the images
    left_frame_pts = []
    right_frame_pts = []

    for match in matches:
        left_frame_pts.append(left_frame_keypoints[match[0].queryIdx].pt)
        right_frame_pts.append(right_frame_keypoints[match[0].trainIdx].pt)

    # Changing the datatype to "float32" for finding homography
    left_frame_pts = np.float32(left_frame_pts)
    right_frame_pts = np.float32(right_frame_pts)

    # Finding the homography matrix(transformation matrix).
    homography_matrix, _ = cv2.findHomography(srcPoints=right_frame_pts, dstPoints=left_frame_pts, method=cv2.LMEDS, ransacReprojThreshold=ransacReprojThreshold)

    return homography_matrix
    
def get_new_frame_size_and_matrix(homography_matrix: np.ndarray, left_frame_shape: tuple[int, int], right_frame_shape: tuple[int, int]) -> tuple[list[list], list[int], np.ndarray]:
    
    # Reading the size of the image
    height, width = left_frame_shape
    
    # Taking the matrix of initial coordinates of the corners of the secondary image
    # Stored in the following format: [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]
    # Where (xi, yi) is the coordinate of the i th corner of the image. 
    initial_matrix = np.array([[0, width - 1, width - 1, 0], [0, 0, height - 1, height - 1], [1, 1, 1, 1]])
    
    # Finding the final coordinates of the corners of the image after transformation.
    # NOTE: Here, the coordinates of the corners of the frame may go out of the 
    # frame(negative values). We will correct this afterwards by updating the 
    # homography matrix accordingly.
    final_matrix = np.dot(a=homography_matrix, b=initial_matrix)

    [x, y, c] = final_matrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Finding the dimentions of the stitched image frame and the "correction" factor
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    new_width = max_x
    new_height = max_y
    correction = [0, 0]

    if min_x < 0:
        new_width -= min_x
        correction[0] = abs(min_x)

    if min_y < 0:
        new_height -= min_y
        correction[1] = abs(min_y)
    
    # Again correcting new_width and new_height
    # Helpful when secondary image is overlaped on the left hand side of the Base image.
    if new_width < right_frame_shape[1] + correction[0]:
        new_width = right_frame_shape[1] + correction[0]

    if new_height < right_frame_shape[0] + correction[1]:
        new_height = right_frame_shape[0] + correction[1]

    # Finding the coordinates of the corners of the image if they all were within the frame.
    x = np.add(x, correction[0])
    y = np.add(y, correction[1])
    old_initial_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    new_final_points = np.float32(np.array([x, y]).transpose())

    # Updating the homography matrix. Done so that now the secondary image completely lies inside the frame
    homography_matrix = cv2.getPerspectiveTransform(old_initial_points, new_final_points)
    
    return [new_height, new_width], correction, homography_matrix

def stitch_images(
        left_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, 
        right_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, 
        value: float = 0.99, 
        angle: float = 2, 
        k: int = 2, 
        ransacReprojThreshold: float = 4,
        clear_cache: bool = True, 
        f_matches: bool = True) -> tuple[np.ndarray, np.ndarray | None]:

    # Cache
    function = eval(inspect.stack()[0][3])

    try:
        new_frame_size, correction, homography_matrix = function.cache

        if clear_cache:
            new_frame_size, correction, homography_matrix = None, None, None

    except:
        new_frame_size, correction, homography_matrix = None, None, None

    # If no cache or clear_cache is set to True, recalculate everything
    if all(obj is None for obj in [new_frame_size, correction, homography_matrix]):

        # Finding matches between the 2 images and their keypoints
        matches, left_frame_keypoints, right_frame_keypoints = find_matches(left_frame=left_frame, right_frame=right_frame, k=k)
        
        # Filter matches
        matches = filter_matches(matches=matches, left_frame_keypoints=left_frame_keypoints, right_frame_keypoints=right_frame_keypoints, value=value, angle=angle)
        
        # Finding homography matrix
        homography_matrix = find_homography(matches=matches, left_frame_keypoints=left_frame_keypoints, right_frame_keypoints=right_frame_keypoints, ransacReprojThreshold=ransacReprojThreshold)
        
        # Finding size of new frame of stitched images and updating the homography matrix
        new_frame_size, correction, homography_matrix = get_new_frame_size_and_matrix(homography_matrix, right_frame.shape[:2], left_frame.shape[:2])

        function.cache = new_frame_size, correction, homography_matrix

    # Finally placing the images upon one another
    stitched_image = cv2.warpPerspective(right_frame, homography_matrix, (new_frame_size[1], new_frame_size[0]))
    stitched_image[correction[1]:correction[1]+left_frame.shape[0], correction[0]:correction[0]+left_frame.shape[1]] = left_frame
    
    # If specified, draw matches
    if f_matches:
        frame_matches = cv2.drawMatchesKnn(
            img1=left_frame, 
            keypoints1=left_frame_keypoints, 
            img2=right_frame, 
            keypoints2=right_frame_keypoints, 
            matches1to2=matches, 
            outImg=None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, 
            matchColor=(0, 0, 255), singlePointColor=(0, 255, 255))
    else:
        frame_matches = None

    return stitched_image, frame_matches