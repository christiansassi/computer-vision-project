import cv2
import inspect
import numpy as np
from shapely.geometry import Polygon, LineString

def _limit_detection(contours: tuple) -> tuple:

    #! TO BE ADJUSTED WITH THE FINAL STITCHED IMAGE
    a = (316, 96)
    b = (1410, 57)
    c = (1516, 1080)
    d = (367, 1124)

    intercepted_contours = []

    polygon = Polygon(np.array([a, b, c, d]))

    for contour in contours:

        if polygon.intersects(LineString(contour.squeeze())):
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
    ref_frame_gray = cv2.GaussianBlur(ref_frame_gray, (5, 5), 0)

    # Convert current frame to gray and apply gaussian blur
    frame = mat.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    # Calculate abs difference between the two frames
    diff = cv2.absdiff(ref_frame_gray, frame_gray)

    # Apply a threshold to get the binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=10)

    # Extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _limit_detection(contours=contours)

    bounding_boxes = []

    # Draw bounding boxes around detected motion
    for contour in contours:
        
        # Ignore small areas
        if cv2.contourArea(contour) < 1500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Update based on specified window
    function.time_window = function.time_window + 1

    if function.time_window == time_window:
        function.ref_frame = frame
        function.time_window = 0

    return original_frame, bounding_boxes

def background_substraction(background: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat) -> tuple[np.ndarray, list[tuple]]:

    # Copy the original frame
    original_frame = mat.copy()

    # Convert background to gray and apply gaussian blur
    _background = background.copy()
    background_gray = cv2.cvtColor(_background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (5, 5), 0)

    # Convert frame to gray and apply gaussian blur
    frame = mat.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    # Calculate abs difference between the two frames
    diff = cv2.absdiff(background_gray, frame_gray)

    # Apply a threshold to get the binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _limit_detection(contours=contours)

    bounding_boxes = []

    # Draw bounding boxes around detected motion
    for contour in contours:
        
        # Ignore small areas
        if cv2.contourArea(contour) < 3000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return original_frame, bounding_boxes

def adaptive_background_substraction(background: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, alpha: float) -> tuple[np.ndarray, list[tuple]]:

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
    background_gray = cv2.GaussianBlur(background_gray, (5, 5), 0)

    # Convert frame to gray and apply gaussian blur
    frame = mat.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    # Calculate abs difference between the two frames
    diff = cv2.absdiff(background_gray, frame_gray)

    # Apply a threshold to get the binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _limit_detection(contours=contours)

    bounding_boxes = []

    # Draw bounding boxes around detected motion
    for contour in contours:
        
        # Ignore small areas
        if cv2.contourArea(contour) < 2000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    function.background = cv2.addWeighted(frame, alpha, function.background, 1 - alpha, 0)

    return original_frame, bounding_boxes
