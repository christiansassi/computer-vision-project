from os import listdir, mkdir
from os.path import join, exists, isfile, basename
import sys

import logging

import cv2

from src import cut_video
from src import stitch_image
from src import utils
from src import params
from src import blending

from src import wrapped_logging_handler

# Levels used: DEBUG, INFO
logger = logging.getLogger()

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

def _stitch_video(videos: list[str], live: bool = True) -> None:

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
            div_left = params.TOP_DIV_LEFT
            div_right = params.TOP_DIV_RIGHT
            frame_number = params.TOP_FRAME
            left_width = params.TOP_COMMON_LEFT
            right_width = params.TOP_COMMON_RIGHT
            intersection = params.TOP_INTERSECTION

        elif "center" in video:
            div_left = params.CENTER_DIV_LEFT
            div_right = params.CENTER_DIV_RIGHT
            frame_number = params.CENTER_FRAME
            left_width = params.CENTER_COMMON_LEFT
            right_width = params.CENTER_COMMON_RIGHT
            intersection = params.CENTER_INTERSECTION

        elif "bottom" in video:
            div_left = params.BOTTOM_DIV_LEFT
            div_right = params.BOTTOM_DIV_RIGHT
            frame_number = params.BOTTOM_FRAME
            left_width = params.BOTTOM_COMMON_LEFT
            right_width = params.BOTTOM_COMMON_RIGHT
            intersection = params.BOTTOM_INTERSECTION

        else:
            raise Exception("Unknwon video")

        value = params.VALUE
        angle = params.ANGLE

        # Pre-process the selected frame and cache the results
        frame = utils.extract_frame(video=video, frame_number=frame_number)
        left_frame, right_frame = utils.split_frame(mat=frame, div_left=div_left, div_right=div_right)
        left_frame, right_frame = utils.black_box_on_image(left_frame=left_frame, right_frame=right_frame, left_width=left_width, right_width=right_width)
        stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle)

        # Open video
        video = cv2.VideoCapture(video)
        assert video.isOpened(), "An error occours while reading the video"

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if params.FRAMES_DEMO is None else params.FRAMES_DEMO
        fps = int(video.get(cv2.CAP_PROP_FPS))
        processed_frames = []

        while True:
            
            # Extract frame by frame
            success, frame = video.read()

            if not success:
                break

            frame = frame[:, div_left:div_right+1]
            left_frame = frame[:, 0:frame.shape[1]//2]
            right_frame = frame[:, frame.shape[1]//2:]

            # Stitch frame
            frame, _ = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, clear_cache=False, f_matches=False)
            video_name = video_name.replace(".mp4", "")
            cv2.imwrite(f"videos/blend/not_blend_{video_name}.jpg", frame)
            # Blend frame
            frame = blending.blend_image(mat=frame, intersection=intersection, intensity=3)
            cv2.imwrite(f"videos/blend/blend_{video_name}.jpg", frame)

            # Auto resize the extracted frame
            frame = utils.auto_resize(mat=frame, ratio=1)

            if live:
                # Display the processed frame
                cv2.imshow(winname="", mat=frame)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

            # Save the processed frame
            processed_frames.append(frame)

            logger.info(f"Processing {video_name}: {int(len(processed_frames) * 100 / frames)}% ({len(processed_frames)} / {frames})\r")
            sys.stdout.flush()

            if len(processed_frames) == frames:
                break
        
        if live:
            cv2.destroyAllWindows()

        output_video = join(processed_videos_folder, video_name)
        
        logging.info(f"Saving {video_name} to {output_video}...\n")

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

if __name__ == "__main__":

    # Setup logger
    logger.setLevel(logging.INFO)
    
    handler = wrapped_logging_handler.WrappedLoggingHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)

    #? Cut video (just once)
    videos = _cut_video()

    #? Stitch video
    _stitch_video(videos=videos)

    #? Detection

    #? Tracking

    #? Team identification

    #? Ball tracking

    # Cleanup logger
    logger.removeHandler(handler)