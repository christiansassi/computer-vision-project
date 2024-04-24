from os import listdir, mkdir
from os.path import join, exists, isfile, basename

import cv2
import numpy as np

from src import cut_video
from src import stitch_image
from src import utils
from src import params
from src import field_extraction

def _cut_video() -> list[str]:

    # Check if cut videos already exist. If not create the workspace
    cut_videos_folder = params.CUT_VIDEOS_FOLDER

    if exists(cut_videos_folder):
        cut_videos = [join(cut_videos_folder, f) for f in listdir(cut_videos_folder) if f.endswith(".mp4")]
    else:
        mkdir(cut_videos_folder)
        cut_videos = []

    if not len(cut_videos):
        original_video_folder = params.ORIGINAL_VIDEOS_FOLDER
        original_videos = [f for f in listdir(original_video_folder) if f.endswith(".mp4")]

        for input_video in original_videos:
            output_video = join(cut_videos_folder, input_video)
            input_video = join(original_video_folder, input_video)
            
            # Cut video
            videos = cut_video.cut(input_video=input_video, output_video=output_video, t1=30)
    else:
        videos = cut_videos
    
    return videos

def _stitch_video(videos: list[str]) -> None:

    processed_videos_folder = params.PROCESSED_VIDEOS_FOLDER

    if not exists(processed_videos_folder):
        mkdir(processed_videos_folder)

    # Process each video
    for video in sorted(videos):
        assert all(isfile(video) for video in videos), f"Unable to locate {video}"
        
        # Save the video name for later
        video_name = basename(video)

        # Some frames give best keypoints and descriptors.
        # We use one of them to cache the homography_matrix and the associated parameters in order to use them later
        if "top" in video:
            value = params.TOP_VALUE
            div_left = params.TOP_DIV_LEFT
            div_right = params.TOP_DIV_RIGHT
            frame_number = params.FRAME_NUMBER_TOP
            video_name = "top"

        elif "center" in video:
            value = params.CENTER_VALUE
            div_left = params.CENTER_DIV_LEFT
            div_right = params.CENTER_DIV_RIGHT
            frame_number = params.FRAME_NUMBER_CENTER
            video_name = "center"

        elif "bottom" in video:
            value = params.BOTTOM_VALUE
            div_left = params.BOTTOM_DIV_LEFT
            div_right = params.BOTTOM_DIV_RIGHT
            frame_number = params.FRAME_NUMBER_BOTTOM
            video_name = "bottom"

        else:
            raise Exception("Unknwon video")

        # Pre-process the selected frame and cache the results
        left_frame, right_frame = utils.extract_frame(video=video, div_left=div_left, div_right=div_right, frame_number=frame_number)
        lf = left_frame.copy()
        rf = right_frame.copy()

        left_frame, right_frame = utils.bb_on_image(left_frame=left_frame, right_frame=right_frame)

        # Open video
        video = cv2.VideoCapture(video)
        assert video.isOpened(), "An error occours while reading the video"

        # _, right_field_mask = field_extraction.extract(mat=right_frame, side=field_extraction.Side.RIGHT, margin=params.MARGIN)
        # _, left_field_mask = field_extraction.extract(mat=left_frame, side=field_extraction.Side.LEFT, margin=params.MARGIN)

        _, frame_matches = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value)
        frame_matches = utils.auto_resize(mat=frame_matches, ratio=1.5)
        utils.show_img(frame_matches, f"Matches_{video_name}")

        frame, _ = stitch_image.stitch_images(left_frame=lf, right_frame=rf, value=value, clear_cache=False, f_matches=False)
        frame = utils.auto_resize(mat=frame, ratio=1.5)
        utils.show_img(frame, "Frame")

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if params.FRAMES_DEMO is None else params.FRAMES_DEMO
        fps = int(video.get(cv2.CAP_PROP_FPS))
        processed_frames = []

        while True:

            success, frame = video.read()

            if not success:
                break

            # Auto resize the extracted frame
            frame = frame[:, div_left:div_right+1]

            left_frame = frame[:, 0:frame.shape[1]//2]
            # left_frame[left_field_mask == False] = (0,0,0)

            right_frame = frame[:, frame.shape[1]//2:]
            # right_frame[right_field_mask == False] = (0,0,0)

            # Stitch frame
            frame, _ = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, clear_cache=False, f_matches=False)
            frame = utils.auto_resize(mat=frame)

            processed_frames.append(frame)

            print(f"Processing {video_name}: {int(len(processed_frames) * 100 / frames)}% ({len(processed_frames)} / {frames})", end="\r")

            if len(processed_frames) == frames:
                break

            # Display the processed frame
            frame = utils.auto_resize(mat=frame, ratio=1.5)
            cv2.imshow("", frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        print("")

        output_video = join(processed_videos_folder, video_name)

        print(f"Saving {video_name} to {output_video}...",end="")

        # Save the processed video
        frame_height, frame_width, _ = processed_frames[0].shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

        # Write frames to the video
        for frame in processed_frames:
            out.write(frame)

        # Release the VideoWriter object
        out.release()
        video.release()

        print("DONE")

if __name__ == "__main__":

    #? Cut video (just once)
    videos = _cut_video()

    #? Stitch video
    _stitch_video(videos=videos)

    #? Detection

    #? Tracking

    #? Team identification

    #? Ball tracking