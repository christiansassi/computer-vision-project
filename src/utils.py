import cv2
import numpy as np
import screeninfo
from uuid import uuid4
from typing import Union
from os.path import isfile
from copy import deepcopy

def auto_resize(mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, ratio: float = 2) -> np.ndarray:

    # Copy the image
    _mat = mat.copy()

    # Get monitor info in order to calculate the best size for the image
    monitors = screeninfo.get_monitors()

    width = min(monitors, key=lambda monitor: monitor.width).width
    height = min(monitors, key=lambda monitor: monitor.height).height

    if height < width:
        # Calculate new height
        height = height // ratio

        # Calculate new width proportional to the height
        width = height * _mat.shape[1] // _mat.shape[0]
    
    else:
        # Calculate new width
        width = width // ratio

        # Calculate new height proportional to the width
        height = width * _mat.shape[0] // _mat.shape[1]

    winsize = (int(width), int(height))

    # Resize the image
    _mat = cv2.resize(_mat, winsize)

    return _mat

def show_img(mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat | list[cv2.typing.MatLike] | list[cv2.cuda.GpuMat] | list[cv2.UMat], winname: str | list[str] = "", ratio: float = 2) -> None:

    # Prepare image(s) and label(s)
    if not isinstance(mat, list):
        mat = [mat]
        winname = [str(winname)]
    
    else:
        if not isinstance(winname, list):
            winname = [str(uuid4()) for _ in range(len(mat))]
        else:
            assert len(winname) == len(mat), "winname must have the same length of mat"
            assert len(winname) == len(set(winname)), "winnames must be unique"

    for i in range(len(mat)):

        # Process the image
        tmp = auto_resize(mat=mat[i], ratio=ratio)

        # Show the image
        cv2.imshow(winname=winname[i], mat=tmp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_frame(video: str | cv2.VideoCapture, frame_number: int) -> np.ndarray:

    if isinstance(video, str):

        assert isfile(video), f"'{video}' is not a valid video"

        video_capture = cv2.VideoCapture(video)
        assert video_capture.isOpened(), "An error occours while opening the video"

        video_frame_bak = None

    elif isinstance(video, cv2.VideoCapture):
        
        video_capture = video
        video_frame_bak = video_capture.get(cv2.CAP_PROP_POS_FRAMES)

    else:
        raise Exception("Invalid video type")

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video_capture.read()

    assert success, "Could not extract the selected frame"

    if video_frame_bak is not None:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame_bak)

    return frame

def split_frame(mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, div_left: int, div_right: int) -> tuple[np.ndarray, np.ndarray]:

    _mat = mat.copy()

    frame = _mat[:, div_left:div_right+1]
    left_frame = frame[:, 0:frame.shape[1]//2]
    right_frame = frame[:, frame.shape[1]//2:]
    
    return left_frame, right_frame

def black_box_on_image(left_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, right_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, left_width: int = None, left_height: int = None, right_width: int = None, right_height: int = None) -> tuple[np.ndarray, np.ndarray]:

    # Adjust widths and heights if needed
    if left_width is None or left_width > left_frame.shape[1]:
        left_width = left_frame.shape[1] - 1

    if left_height is None or left_height > left_frame.shape[0]:
        left_height = left_frame.shape[0] - 1
    
    if right_width is None or right_width > right_frame.shape[1]:
        right_width = right_frame.shape[1] - 1

    if right_height is None or right_height > right_frame.shape[0]:
        right_height = right_frame.shape[0] - 1

    # Left frame
    x, y = 0, 0
    left_frame[y:y + left_height + 1, x:x + left_width + 1] = tuple([0 for _ in range(left_frame[0, 0].shape[0])])
    
    # Right frame
    x, y = right_width, 0
    right_frame[y:y + right_height + 1, x:] = tuple([0 for _ in range(right_frame[0, 0].shape[0])])

    return left_frame, right_frame

def crop_image(image: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat) -> np.ndarray:

    height, width = image.shape[:2]

    center_x = width // 2
    center_y = height // 2

    image = image[center_y - 878:center_y + 879, center_x - 878:center_x + 879]

    return image

def bb(left_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, right_frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, left_min: int = None, left_max: int = None, right_min: int = None, right_max: int = None) -> tuple[np.ndarray, np.ndarray]:
    
    _left_frame = left_frame.copy()
    _right_frame = right_frame.copy()

    _left_frame[:, :left_min] = 0
    _left_frame[:, left_max:] = 0
    
    _right_frame[:, :right_min] = 0
    _right_frame[:, right_max:] = 0

    return _left_frame, _right_frame

def jpg_compression(mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat) -> np.ndarray:
    return cv2.imdecode(cv2.imencode('.jpg', mat, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1], cv2.IMREAD_UNCHANGED)