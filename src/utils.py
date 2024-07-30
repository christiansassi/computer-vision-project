import cv2
import numpy as np
import screeninfo
from uuid import uuid4
from typing import Union
from os.path import isfile

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

    winsize = (width, height)

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

def extract_frame(video: str, frame_number: int) -> np.ndarray:

    assert isfile(video), f"'{video}' is not a valid video"

    video = cv2.VideoCapture(video)
    assert video.isOpened(), "An error occours while reading the video"

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video.read()

    assert success, "Could not extract the selected frame"

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