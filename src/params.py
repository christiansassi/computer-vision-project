
#? CONST

VALUE = 0.99 # Value for matching keypoints
ANGLE = 2 # Angle for matching keypoints
FRAMES_DEMO = None # Limit for demo test. None = no demo
BACKGROUND_FRAME = 5376

TOP = {
    "div_left": 899,
    "div_right": 3194,
    "left_width": 913,
    "right_width": 231, 
    "frame_number": 120,
    "intersection": 1147
}

CENTER = {
    "div_left": 1024,
    "div_right": 3072,
    "left_width": 884,
    "right_width": 144, 
    "frame_number": 120,
    "intersection": 1024
}

BOTTOM = {
    "div_left": 899,
    "div_right": 3194,
    "left_width": 786,
    "right_width": 370, 
    "frame_number": 120,
    "intersection": 1148
}

#? USER

from os.path import join

ROOT = "videos"

ORIGINAL_VIDEOS_FOLDER = join(ROOT, "original")
CUT_VIDEOS_FOLDER = join(ROOT, "cut")
STITCHED_VIDEOS_FOLDER = join(ROOT, "stitched")