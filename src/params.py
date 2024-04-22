
#? CONST

TOP_VALUE = 0.4 # Value for matching keypoints
TOP_DIV_LEFT = 899 # Left div
TOP_DIV_RIGHT = 3194 # Right div

BOTTOM_VALUE = 0.57 # Value used for matching keypoints
BOTTOM_DIV_LEFT = TOP_DIV_LEFT # Left div
BOTTOM_DIV_RIGHT = TOP_DIV_RIGHT # Right div

CENTER_VALUE = 0.5 # Value used for matching keypoints
CENTER_DIV_LEFT = 1024 # Left div
CENTER_DIV_RIGHT = 3072 # Right div

FRAME_NUMBER_TOP = 1000 # Reference for top frame
FRAME_NUMBER_BOTTOM = 1000 # Reference for bottom frame
FRAME_NUMBER_CENTER = 1000 # Reference for center frame

MARGIN = 100 # Margin volleyball field

FRAMES_DEMO = None # Limit for demo test. None = no demo

#? USER

ORIGINAL_VIDEOS_FOLDER = r"videos/original"
CUT_VIDEOS_FOLDER = r"videos/cut"
PROCESSED_VIDEOS_FOLDER = r"videos/processed"