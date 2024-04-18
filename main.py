from os import listdir, mkdir
from os.path import join, exists, isfile

import cv2

from src import cut_video
from src import stitch_image
from src import utils
from src import params

if __name__ == "__main__":

    #? Cut video (just once)

    # Check if cut videos already exist. If not create the workspace
    cut_videos_folder = r"videos\cut"

    if exists(cut_videos_folder):
        cut_videos = [join(cut_videos_folder, f) for f in listdir(cut_videos_folder) if f.endswith(".mp4")]
    else:
        mkdir(cut_videos_folder)
        cut_videos = []

    if not len(cut_videos):
        original_video_folder = r"videos\original"
        original_videos = [f for f in listdir(original_video_folder) if f.endswith(".mp4")]

        for input_video in original_videos:
            output_video = join(cut_videos_folder, input_video)
            input_video = join(original_video_folder, input_video)
            
            # Cut video
            videos = cut_video.cut(input_video=input_video, output_video=output_video, t1=30)
    else:
        videos = cut_videos

    #? Stitch video

    # Process each video
    for video in videos:
        assert all(isfile(video) for video in videos), f"Unable to locate {video}"
        
        value = 0.4
    
        # We noticed that the 500th frame of the top view was the best one in terms of keypoints and descriptors.
        # We use it to cache the homography_matrix and the associated parameters in order to use them later
        if "top" in video:
            left_frame = cv2.imread(r"videos\ref\top_left.png")
            right_frame = cv2.imread(r"videos\ref\top_right.png")
            value = 0.4
            stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value)

        # Open video
        video = cv2.VideoCapture(video)

        assert video.isOpened(), "An error occours while reading the video"

        while True:

            success, frame = video.read()

            if not success:
                break
            
            # Auto resize the extracted frame
            frame = frame[:, params.DIV_LEFT:params.DIV_RIGHT+1]
            left_frame = frame[:, 0:frame.shape[1]//2]
            right_frame = frame[:, frame.shape[1]//2:]

            # Stitch frame
            frame = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, clear_cache=False)
            frame = utils.auto_resize(mat=frame)

            # Display the processed frame
            cv2.imshow("", frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    #? Detection

    #? Tracking

    #? Team identification

    #? Ball tracking