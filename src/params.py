
#? CONST

TOP_VALUE = 0.99 # Value for matching keypoints
TOP_ANGLE = 2 # Angle for matching keypoints
TOP_DIV_LEFT = 899 # Left div
TOP_DIV_RIGHT = 3194 # Right div

CENTER_VALUE = 0.99 # Value used for matching keypoints
CENTER_ANGLE = 2 # Angle for matching keypoints
CENTER_DIV_LEFT = 1024 # Left div
CENTER_DIV_RIGHT = 3072 # Right div

BOTTOM_VALUE = 0.99 # Value used for matching keypoints
BOTTOM_ANGLE = 2 # Angle for matching keypoints
BOTTOM_DIV_LEFT = TOP_DIV_LEFT # Left div
BOTTOM_DIV_RIGHT = TOP_DIV_RIGHT # Right div

FRAME_NUMBER_TOP = 120 # Reference for top frame
FRAME_NUMBER_BOTTOM = 120 # Reference for bottom frame
FRAME_NUMBER_CENTER = 120 # Reference for center frame

MARGIN = 100 # Margin volleyball field

FRAMES_DEMO = None # Limit for demo test. None = no demo

#? USER
from os.path import join

ROOT = "videos"

ORIGINAL_VIDEOS_FOLDER = join(ROOT, "original")
CUT_VIDEOS_FOLDER = join(ROOT, "cut")
PROCESSED_VIDEOS_FOLDER = join(ROOT, "processed")