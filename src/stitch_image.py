import cv2
import numpy as np  
import math
import logging

def __filter_matches(matches: list[list], left_frame_keypoints: tuple[cv2.KeyPoint], right_frame_keypoints: tuple[cv2.KeyPoint], value: float, angle: float) -> list[list]:
    # Filter matches based on angles
    _matches = []
    
    # Loop through all the matches
    for index, match in enumerate(matches):
        fp_x, fp_y = left_frame_keypoints[match[0].queryIdx].pt
        sp_x, sp_y = right_frame_keypoints[match[0].trainIdx].pt

        # Finding the inclination of the line joining the two points
        inclination = math.degrees(math.atan((sp_y - fp_y)/(sp_x - fp_x)))
        # inclination = math.degrees(math.atan2((sp_y - fp_y), (sp_x - fp_x)))

        # Filtering out the matches based on the inclination
        if isinstance(angle, tuple):
            if angle[0] <= inclination <= angle[1]:
                logging.debug(f"Match {index} with inclination: {inclination} and distance {match[0].distance} [KEPT]\n")
                _matches.append(match)
            else:
                logging.debug(f"Match {index} with inclination: {inclination} and distance {match[0].distance} [REMOVED]\n")
        else:
            if abs(inclination) <= angle:
                logging.debug(f"Match {index} with inclination: {inclination} and distance {match[0].distance} [KEPT]\n")
                _matches.append(match)
            else:
                logging.debug(f"Match {index} with inclination: {inclination} and distance {match[0].distance} [REMOVED]\n")

    # Applying ratio test and filtering out the good matches
    _matches = [[match1] for match1, match2 in _matches if match1.distance < value * match2.distance]

    return _matches

def __find_matches(left_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, right_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, k: int) -> tuple[list[list], tuple[cv2.KeyPoint], tuple[cv2.KeyPoint]]:

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

def __find_homography(matches: list[list], left_frame_keypoints: tuple[cv2.KeyPoint], right_frame_keypoints: tuple[cv2.KeyPoint], ransacReprojThreshold: float, method: int) -> np.ndarray:

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
    homography_matrix, _ = cv2.findHomography(srcPoints=right_frame_pts, dstPoints=left_frame_pts, method=method, ransacReprojThreshold=ransacReprojThreshold)

    return homography_matrix
    
def __get_new_frame_size_and_matrix(homography_matrix: np.ndarray, left_frame_shape: tuple[int, int], right_frame_shape: tuple[int, int]) -> tuple[list[list], list[int], np.ndarray]:
    
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
        # Required
        left_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, 
        right_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, 

        # Ok
        value: float = 0.99, 
        angle: float = 2, 

        # Not required
        k: int = 2, 
        ransacReprojThreshold: float = 4,

        # Required
        method: int = cv2.LMEDS,

        # Stitching params
        new_frame_size = None,
        correction = None,
        homography_matrix = None,

        # Stitching keypoints
        user_left_kp: list = None,
        user_right_kp: list = None,

        # Tuning
        left_shift_dx: int = 0,
        left_shift_dy: int = 0,
        remove_offset: int = 0,
        
        # Debug
        f_matches: bool = False) -> tuple[np.ndarray, np.ndarray | None, tuple]:
    
    # If no params, recalculate everything
    if all(obj is None for obj in [new_frame_size, correction, homography_matrix]):

        # Finding matches between the 2 images and their keypoints
        matches, left_frame_keypoints, right_frame_keypoints = __find_matches(left_frame=left_frame, right_frame=right_frame, k=k)
        
        # Filter matches
        matches = __filter_matches(matches=matches, left_frame_keypoints=left_frame_keypoints, right_frame_keypoints=right_frame_keypoints, value=value, angle=angle)

        if user_left_kp is not None and user_right_kp is not None:
            left_frame_keypoints = list(left_frame_keypoints)
            right_frame_keypoints = list(right_frame_keypoints)
            

            for i in range(len(user_left_kp)):
                left_frame_keypoints.append(cv2.KeyPoint(x=user_left_kp[i][0], y=user_left_kp[i][1], size=1))
                right_frame_keypoints.append(cv2.KeyPoint(x=user_right_kp[i][0], y=user_right_kp[i][1], size=1))
            
            left_frame_keypoints = tuple(left_frame_keypoints)
            right_frame_keypoints = tuple(right_frame_keypoints)
                                        
            # Manually adding some matches
            # [queryIdx, trainIdx, distance]
            manual_matches = []
            for i in range(len(user_left_kp)):
                manual_matches.append([cv2.DMatch(len(left_frame_keypoints)-i-1, len(right_frame_keypoints)-i-1, 0)])

            matches.extend(manual_matches)

        # Finding homography matrix
        homography_matrix = __find_homography(matches=matches, left_frame_keypoints=left_frame_keypoints, right_frame_keypoints=right_frame_keypoints, ransacReprojThreshold=ransacReprojThreshold, method=method)
        
        # Finding size of new frame of stitched images and updating the homography matrix
        new_frame_size, correction, homography_matrix = __get_new_frame_size_and_matrix(homography_matrix, right_frame.shape[:2], left_frame.shape[:2])

    parameters = (new_frame_size, correction, homography_matrix)

    # Finally placing the images upon one another
    stitched_image = cv2.warpPerspective(right_frame, homography_matrix, (new_frame_size[1], new_frame_size[0]))

    # Determine the region where the left frame will be placed
    region_x_start = left_shift_dx + correction[0]
    region_x_end = correction[0] + left_frame.shape[1] + left_shift_dx - remove_offset
    region_y_start = correction[1] + left_shift_dy
    region_y_end = correction[1] + left_frame.shape[0] + left_shift_dy

    # Ensure the regions are within bounds
    region_x_end = min(region_x_end, stitched_image.shape[1])
    region_y_end = min(region_y_end, stitched_image.shape[0])

    # Place the left frame into the stitched image
    stitched_image[region_y_start:region_y_end, region_x_start:region_x_end] = left_frame[:region_y_end-region_y_start, :region_x_end-region_x_start]

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

    return stitched_image, frame_matches, parameters