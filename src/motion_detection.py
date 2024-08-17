import cv2
import inspect
import numpy as np
from shapely.geometry import Polygon, LineString
from src import utils

FRAME_SUBSTRACTION: int = 1
BACKGROUND_SUBSTRACTION: int = 2
ADAPTIVE_BACKGROUND_SUBSTRACTION: int = 3

def _filter_contours(contours: tuple, min_contour_area: int) -> tuple:

    #! TO BE ADJUSTED WITH THE FINAL STITCHED IMAGE
    a = (34, 62)
    b = (2308, 78)
    c = (2232, 1332)
    d = (48, 1254)

    intercepted_contours = []

    polygon = Polygon(np.array([a, b, c, d]))
    polygon = polygon.buffer(100)

    for contour in contours:
        
        # Ignore small areas
        if cv2.contourArea(contour) < min_contour_area:
            continue
        
        # Ignore non-intersecting contours 
        if not polygon.intersects(LineString(contour.squeeze())):
            continue
        
        intercepted_contours.append(contour)

    return tuple(intercepted_contours)

def frame_substraction(mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, time_window: int = 1) -> tuple[np.ndarray, list[tuple]]:

    # Copy the original frame
    original_frame = mat.copy()

    # Store the previous frame
    function = eval(inspect.stack()[0][3])

    try:
        function.ref_frame
        function.time_window
    
    except:
        function.ref_frame = mat.copy()
        function.time_window = 0

    # Convert ref frame to gray and apply gaussian blur
    ref_frame = function.ref_frame
    ref_frame_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_frame_gray = cv2.GaussianBlur(ref_frame_gray, (15, 15), 0)

    # Convert current frame to gray and apply gaussian blur
    frame = mat.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)

    # Calculate abs difference between the two frames
    diff = cv2.absdiff(ref_frame_gray, frame_gray)

    # Apply a threshold to get the binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _filter_contours(contours=contours, min_contour_area=6500)

    bounding_boxes = []

    # Draw bounding boxes around detected motion
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Update based on specified window
    function.time_window = function.time_window + 1

    if function.time_window == time_window:
        function.ref_frame = frame
        function.time_window = 0

    return original_frame, bounding_boxes

def background_substraction(mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, background: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat) -> tuple[np.ndarray, list[tuple]]:

    # Copy the original frame
    original_frame = mat.copy()

    # Convert background to gray and apply gaussian blur
    _background = background.copy()
    background_gray = cv2.cvtColor(_background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (15, 15), 0)
    
    # Convert frame to gray and apply gaussian blur
    frame = mat.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
    
    #cv2.imwrite("blur.jpg", frame_gray)
    #exit(0)

    # Calculate abs difference between the two frames
    diff = cv2.absdiff(background_gray, frame_gray)

    # Apply a threshold to get the binary image
    _, thresh = cv2.threshold(diff, 13, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _filter_contours(contours=contours, min_contour_area=4000)

    bounding_boxes = []

    # Draw bounding boxes around detected motion
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return original_frame, bounding_boxes, thresh

def adaptive_background_substraction(mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, background: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, alpha: float) -> tuple[np.ndarray, list[tuple]]:

    # Check alpha
    assert alpha >= 0 and alpha <= 1, "Alpha must be a number in the interval [0, 1]"

    # Copy the original frame
    original_frame = mat.copy()

    # Store the background frame so we can update it
    function = eval(inspect.stack()[0][3])

    try:
        function.background
    
    except:
        function.background = background.copy()
    
    # Convert background to gray and apply gaussian blur
    background_gray = cv2.cvtColor(function.background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (15, 15), 0)

    # Convert frame to gray and apply gaussian blur
    frame = mat.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)

    # Calculate abs difference between the two frames
    diff = cv2.absdiff(background_gray, frame_gray)

    # Apply a threshold to get the binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _filter_contours(contours=contours, min_contour_area=4000)

    bounding_boxes = []

    # Draw bounding boxes around detected motion
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    function.background = cv2.addWeighted(frame, alpha, function.background, 1 - alpha, 0)

    return original_frame, bounding_boxes

def detection(frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, detection_type: int, background: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat = None, alpha: float = None) -> tuple[np.ndarray, list[tuple]]:

        if detection_type == FRAME_SUBSTRACTION:

            # Apply frame substraction
            #* PROS
            #* [+] None (for this purpose)

            #! CONS
            #! [-] Stops detecting an object if it stops moving
            #! [-] A larger window can avoid the previous problem but would negatively impact detection quality
            #processed_frame, bounding_boxes = frame_substraction(mat=frame, time_window=7)

            # Apply background substraction
            #* PROS
            #* [+] Good since the background doesn't change too much (for this purpose)
            #* [+] Keeps detecting objects even if they stop moving

            #! CONS
            #! [-] None (for this purpose)
            
            return background_substraction(background=background, mat=frame)

        elif detection_type == BACKGROUND_SUBSTRACTION:
            
            # Apply background substraction
            #* PROS
            #* [+] Good since the background doesn't change too much (for this purpose)
            #* [+] Keeps detecting objects even if they stop moving

            #! CONS
            #! [-] None (for this purpose)

            assert background is not None, "Background not defined"

            return background_substraction(background=background, mat=frame)

        elif detection_type == ADAPTIVE_BACKGROUND_SUBSTRACTION:
            
            # Apply adaptive substraction
            #* PROS
            #* [+] Good for this purpose since the background doesn't change too much
            #* [+] Compared to normal background subtraction, it adapts to small background changes

            #! CONS
            #! [-] A large alpha value causes the algorithm to stop detecting objects that have stopped moving
            #! [-] Since we are forced to use a small alpha value, this algorithm becomes similar to normal background subtraction

            assert background is not None, "Background not defined"
            assert alpha is not None, "Alpha not defined"

            return adaptive_background_substraction(background=background, mat=frame, alpha=alpha)
