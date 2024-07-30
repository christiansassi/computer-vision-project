
#? CONST

VALUE = 0.99 # Value for matching keypoints
ANGLE = 2 # Angle for matching keypoints

TOP_DIV_LEFT = 899 # Left div
TOP_DIV_RIGHT = 3194 # Right div
TOP_COMMON_LEFT = 913 # Leftmost common point
TOP_COMMON_RIGHT = 231 # Rightmost common point
TOP_FRAME = 120 # Reference for top frame
TOP_INTERSECTION = 1147 # Intersection for top frame

CENTER_DIV_LEFT = 1024 # Left div
CENTER_DIV_RIGHT = 3072 # Right div
CENTER_COMMON_LEFT = 884 # Leftmost common point
CENTER_COMMON_RIGHT = 144 # Rightmost common point
CENTER_FRAME = 120 # Reference for center frame
CENTER_INTERSECTION = 1024 # Intersection for center frame

BOTTOM_DIV_LEFT = 899 # Left div
BOTTOM_DIV_RIGHT = 3194 # Right div
BOTTOM_COMMON_LEFT = 786 # Leftmost common point
BOTTOM_COMMON_RIGHT = 370 # Rightmost common point
BOTTOM_FRAME = 120 # Reference for bottom frame
BOTTOM_INTERSECTION = 1148 # Intersection for bottom frame

FRAMES_DEMO = None # Limit for demo test. None = no demo

BACKGROUND_FRAME = 5376

#? USER

from os.path import join

ROOT = "videos"

ORIGINAL_VIDEOS_FOLDER = join(ROOT, "original")
CUT_VIDEOS_FOLDER = join(ROOT, "cut")
PROCESSED_VIDEOS_FOLDER = join(ROOT, "processed")