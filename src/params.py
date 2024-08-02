
#? CONST

VALUE = 0.99 # Value for matching keypoints
ANGLE = 2 # Angle for matching keypoints
FRAMES_DEMO = None # Limit for demo test. None = no demo
BACKGROUND_FRAME = 5376 # Background reference for background motion detection algorithms

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

TOP_CENTER = {
    "left_frame_kp": [(1724, 373), (1704, 606), (1580, 460), (1704, 1126), (1746, 1208), (1680, 1352), (1714, 1462), (1188, 1464)],
    "right_frame_kp": [(1037, 388), (966, 666), (754, 552), (1002, 1224), (1104, 1306), (1006, 1454), (1098, 1560), (402, 1570)],
    "left_shift_dx": 15,
    "left_shift_dy": 15,
    "remove_offset": 780,
    "angle": (-8, -5),
    "value": 0.95,
    "left_min": 950,
    "right_min": 100,
    "right_max": 1100
}

BOTTOM_CENTER = {
    "left_frame_kp": None,  
    "right_frame_kp": None,
    "left_shift_dx": 0,
    "left_shift_dy": 0,
    "remove_offset": 550,
    "angle": (-15, -5),
    "value": 0.95,
    "left_min": 1200,
    "right_min": 100,
    "right_max": 800
}

FINAL = {
    "left_frame_kp": None,
    "right_frame_kp": None, 
    "left_shift_dx": 0,
    "left_shift_dy": 0,
    "remove_offset": 400,
    "angle": (-40, 0),
    "value": 0.95,
    "left_min": 1000,
    "left_max": 1400,
    "right_min": 350,
    "right_max": 770
}

#? USER

from os.path import join

ROOT = "videos"

ORIGINAL_VIDEOS_FOLDER = join(ROOT, "original")
CUT_VIDEOS_FOLDER = join(ROOT, "cut")
STITCHED_VIDEOS_FOLDER = join(ROOT, "stitched")