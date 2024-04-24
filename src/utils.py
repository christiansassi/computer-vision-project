import cv2
import numpy as np
import screeninfo
from uuid import uuid4
from typing import Union

def auto_resize(mat: Union[cv2.typing.MatLike, cv2.cuda.GpuMat, cv2.UMat], ratio: float = 2) -> np.ndarray:

    # Get monitor info in order to calculate the best size for the image(s)
    monitor = screeninfo.get_monitors()[0]

    # Copy the image
    _mat = mat.copy()

    if monitor.height < monitor.width:
        # Calculate new height
        height = int(monitor.height / ratio)

        # Calculate new width proportional to the height
        width = int(height * _mat.shape[1] / _mat.shape[0])
    
    else:
        # Calculate new width
        width = int(monitor.width / ratio)

        # Calculate new height proportional to the width
        height = int(width * _mat.shape[0] / _mat.shape[1])

    winsize = (width, height)

    # Resize the image
    _mat = cv2.resize(_mat, winsize)

    return _mat

def show_img(mat: Union[cv2.typing.MatLike, cv2.cuda.GpuMat, cv2.UMat, list[cv2.typing.MatLike], list[cv2.cuda.GpuMat], list[cv2.UMat]], winname: Union[str, list[str]] = "", ratio: float = 2) -> None:

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

def extract_frame(video: cv2.VideoCapture, div_left: int, div_right: int, frame_number: int = 500) -> tuple[np.ndarray, np.ndarray]:
    video = cv2.VideoCapture(video)
    assert video.isOpened(), "An error occours while reading the video"

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()

    if ret:
        frame = frame[:, div_left:div_right+1]

        left_frame = frame[:, 0:frame.shape[1]//2]
        right_frame = frame[:, frame.shape[1]//2:]
    else:
        raise Exception("Could not extract frame")
    
    return left_frame, right_frame

def bb_on_image(left_frame: Union[cv2.typing.MatLike, cv2.cuda.GpuMat, cv2.UMat], right_frame: Union[cv2.typing.MatLike, cv2.cuda.GpuMat, cv2.UMat]) -> tuple[np.ndarray, np.ndarray]:
    x_spacing = 750
    w, h = x_spacing, left_frame.shape[0]
    
    # Left frame
    x, y = 0, 0 
    black_box = np.zeros_like(left_frame)
    left_frame[y:y+h, x:x+w] = black_box[y:y+h, x:x+w]
    
    # Right frame
    x, y = right_frame.shape[1]-x_spacing, 0
    black_box = np.zeros_like(right_frame)
    right_frame[y:y+h, x:x+w] = black_box[y:y+h, x:x+w]

    return left_frame, right_frame