from os import listdir, mkdir
from os.path import join, exists, isfile, basename
import sys

import logging

import cv2
import numpy as np
import os

from src import cut_video
from src import stitch_image
from src import utils
from src import params
from src import blending
from src import motion_detection

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

    processed_videos_folder = params.STITCHED_VIDEOS_FOLDER

    if not exists(processed_videos_folder):
        mkdir(processed_videos_folder)

    # Process each video
    for video in sorted(videos):
        assert all(isfile(video) for video in videos), f"Unable to locate {video}"
        
        video_name = basename(video)
        video_label = video_name.replace(".mp4", "")

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
        frame, _ = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, method=cv2.LMEDS)
        frame = blending.blend_image(mat=frame, intersection=intersection, intensity=3)
        frame = utils.auto_resize(mat=frame, ratio=1)

        # Open video
        video = cv2.VideoCapture(video)
        assert video.isOpened(), "An error occours while reading the video"

        # Set start at frame with index equals to 0
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if params.FRAMES_DEMO is None else params.FRAMES_DEMO
        processed_frames = 0

        # Create a new video object to save each frame during each interaction
        # Saving all the frames in a list and then saving them is memory-consuming (lots of GBs of RAM)
        # Therefore, this solution is better
        output_video = join(processed_videos_folder, video_name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_height, frame_width, _ = frame.shape
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

        while True:
            
            # Extract frame by frame
            success, frame = video.read()

            if not success:
                break

            frame = frame[:, div_left:div_right+1]
            left_frame = frame[:, 0:frame.shape[1]//2]
            right_frame = frame[:, frame.shape[1]//2:]

            # Stitch frame
            frame, _ = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, method=cv2.LMEDS, clear_cache=False, f_matches=False)

            # Blend frame
            frame = blending.blend_image(mat=frame, intersection=intersection, intensity=3)

            # Save the processed frame
            out.write(frame)

            # Auto resize the extracted frame
            frame = utils.auto_resize(mat=frame, ratio=1.5)

            if live:
                # Display the processed frame
                cv2.imshow(winname="", mat=frame)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

            # Display log info
            processed_frames = processed_frames + 1

            logger.info(f"Processing {video_label} view: {int(processed_frames * 100 / frames)}% ({processed_frames} / {frames})\r")
            sys.stdout.flush()

            if processed_frames == frames:
                break
        
        # Cleanup
        if live:
            cv2.destroyAllWindows()

        out.release()
        video.release()

def _motion_detection(videos: list[str]) -> None:

    for video in sorted(videos):
        
        video_name = video

        # Open video
        video = cv2.VideoCapture(video)
        assert video.isOpened(), "An error occours while reading the video"

        # Set start at frame with index equals to 0
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        background = utils.extract_frame(video=video_name, frame_number=params.BACKGROUND_FRAME)

        while True:

            # Extract frame by frame
            success, frame = video.read()

            if not success:
                break
            
            # Apply frame substraction
            #* PROS
            #* [+] None (for this purpose)

            #! CONS
            #! [-] Stops detecting an object if it stops moving
            #! [-] A larger window can avoid the previous problem but would negatively impact detection quality
            #processed_frame, bounding_boxes = motion_detection.frame_substraction(mat=frame, time_window=7)

            # Apply background substraction
            #* PROS
            #* [+] Good since the background doesn't change too much (for this purpose)
            #* [+] Keeps detecting objects even if they stop moving

            #! CONS
            #! [-] None (for this purpose)
            processed_frame, bounding_boxes = motion_detection.background_substraction(background=background, mat=frame)

            # Apply adaptive substraction
            #* PROS
            #* [+] Good for this purpose since the background doesn't change too much
            #* [+] Compared to normal background subtraction, it adapts to small background changes

            #! CONS
            #! [-] A large alpha value causes the algorithm to stop detecting objects that have stopped moving
            #! [-] Since we are forced to use a small alpha value, this algorithm becomes similar to normal background subtraction
            #processed_frame, bounding_boxes = motion_detection.adaptive_background_substraction(background=background, mat=frame, alpha=0.05)

            processed_frame = utils.auto_resize(mat=processed_frame, ratio=1.5)

            # Display the processed frame
            cv2.imshow(winname="", mat=processed_frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

def _stitch_all_videos(videos: list[str], live: bool = True) -> None:
    
    processed_videos_folder = params.STITCHED_VIDEOS_FOLDER

    if not exists(processed_videos_folder):
        mkdir(processed_videos_folder)

    value = params.VALUE
    angle = params.ANGLE

    # Open all videos
    video_top = None
    new_frame_size_top = None
    correction_top = None
    homography_matrix_top = None

    video_center = None
    new_frame_size_center = None
    correction_center = None
    homography_matrix_center = None

    video_bottom = None
    new_frame_size_bottom = None
    correction_bottom = None
    homography_matrix_bottom = None
    
    for video in videos:

        video_capture = cv2.VideoCapture(video)
        assert video_capture.isOpened(), "An error occours while reading the video"

        if "top" in video:
            video_top = video_capture

            frame = utils.extract_frame(video=video, frame_number=params.TOP["frame_number"])
            left_frame, right_frame = utils.split_frame(mat=frame, div_left=params.TOP["div_left"], div_right=params.TOP["div_right"])
            left_frame, right_frame = utils.black_box_on_image(left_frame=left_frame, right_frame=right_frame, left_width=params.TOP["left_width"], right_width=params.TOP["right_width"])
            _, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, method=cv2.LMEDS)
            new_frame_size_top, correction_top, homography_matrix_top = stitching_params

        elif "center" in video:
            video_center = video_capture

            frame = utils.extract_frame(video=video, frame_number=params.CENTER["frame_number"])
            left_frame, right_frame = utils.split_frame(mat=frame, div_left=params.CENTER["div_left"], div_right=params.CENTER["div_right"])
            left_frame, right_frame = utils.black_box_on_image(left_frame=left_frame, right_frame=right_frame, left_width=params.CENTER["left_width"], right_width=params.CENTER["right_width"])
            _, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, method=cv2.LMEDS)
            new_frame_size_center, correction_center, homography_matrix_center = stitching_params

        elif "bottom" in video:
            video_bottom = video_capture

            frame = utils.extract_frame(video=video, frame_number=params.BOTTOM["frame_number"])
            left_frame, right_frame = utils.split_frame(mat=frame, div_left=params.BOTTOM["div_left"], div_right=params.BOTTOM["div_right"])
            left_frame, right_frame = utils.black_box_on_image(left_frame=left_frame, right_frame=right_frame, left_width=params.BOTTOM["left_width"], right_width=params.BOTTOM["right_width"])
            frame, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, method=cv2.LMEDS)
            new_frame_size_bottom, correction_bottom, homography_matrix_bottom = stitching_params

        else:
            raise Exception("Unknwon video")

    # Load all the images for the final stitching    
    images = []
    for filename in os.listdir('videos/blend'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join('videos/blend', filename)
            img = cv2.imread(image_path)
            images.append(img)

    # Rotate and crop the images
    images = utils.rotate_and_crop(images) # [bottom, top, center_for_top, center_for_bottom]
    cropped_images = {
        "bottom": images[0],
        "top": images[1],
        "center_top": images[2],
        "center_bottom": images[3]
    }

    #* TOP_CENTER
    lf = cropped_images["center_top"].copy()
    rf = cropped_images["top"].copy()

    lf, rf = utils.bb(left_frame=lf, right_frame=rf, left_min=params.TOP_CENTER["left_min"], left_max=lf.shape[1], right_min=params.TOP_CENTER["right_min"], right_max=params.TOP_CENTER["right_max"])

    _, _, stitching_params = stitch_image.stitch_images(left_frame=lf, right_frame=rf, value = params.TOP_CENTER["value"], angle = params.TOP_CENTER["angle"], 
                                                        method = cv2.RANSAC, user_left_kp = params.TOP_CENTER["left_frame_kp"], user_right_kp = params.TOP_CENTER["right_frame_kp"])
    
    new_frame_size_top_center, correction_top_center, homography_matrix_top_center = stitching_params

    #* BOTTOM_CENTER
    lf = cropped_images["center_bottom"].copy()
    rf = cropped_images["bottom"].copy()

    lf, rf = utils.bb(left_frame=lf, right_frame=rf, left_min=params.BOTTOM_CENTER["left_min"], left_max=lf.shape[1], right_min=params.BOTTOM_CENTER["right_min"], right_max=params.BOTTOM_CENTER["right_max"])

    _, _, stitching_params = stitch_image.stitch_images(left_frame=lf, right_frame=rf, value = params.BOTTOM_CENTER["value"], angle = params.BOTTOM_CENTER["angle"],
                                                        method = cv2.RANSAC, user_left_kp = params.BOTTOM_CENTER["left_frame_kp"], user_right_kp = params.BOTTOM_CENTER["right_frame_kp"])
    
    new_frame_size_bottom_center, correction_bottom_center, homography_matrix_bottom_center = stitching_params

    #* FINAL
    

    while True:

        success_top, frame_top = video_top.read()
        success_center, frame_center = video_center.read()
        success_bottom, frame_bottom = video_bottom.read()

        if success_top + success_center + success_bottom != 3:
            break
        
        #! TOP - TOP
        frame_top = frame_top[:, params.TOP["div_left"]:params.TOP["div_right"]+1]
        left_frame_top = frame_top[:, :frame_top.shape[1] // 2] 
        right_frame_top = frame_top[:, frame_top.shape[1] // 2:] 

        # Stitch frame
        frame_top, _, _ = stitch_image.stitch_images(left_frame=left_frame_top, right_frame=right_frame_top, value=value, angle=angle, new_frame_size=new_frame_size_top, correction=correction_top, homography_matrix=homography_matrix_top)

        # Blend frame
        frame_top = blending.blend_image(mat=frame_top, intersection=params.TOP["intersection"], intensity=3)

        # Show frame
        # utils.show_img(frame_top, "TOP", ratio=1.5)

        #! CENTER - CENTER
        frame_center = frame_center[:, params.CENTER["div_left"]:params.CENTER["div_right"]+1]
        left_frame_center = frame_center[:, :frame_center.shape[1] // 2] 
        right_frame_center = frame_center[:, frame_center.shape[1] // 2:] 

        # Stitch frame
        frame_center, _, _ = stitch_image.stitch_images(left_frame=left_frame_center, right_frame=right_frame_center, value=value, angle=angle, new_frame_size=new_frame_size_center, correction=correction_center, homography_matrix=homography_matrix_center)

        # Blend frame
        frame_center = blending.blend_image(mat=frame_center, intersection=params.CENTER["intersection"], intensity=3)

        # Show frame
        # utils.show_img(frame_center, "CENTER", ratio=1.5)

        #! BOTTOM - BOTTOM
        frame_bottom = frame_bottom[:, params.BOTTOM["div_left"]:params.BOTTOM["div_right"]+1]
        left_frame_bottom = frame_bottom[:, :frame_bottom.shape[1] // 2]
        right_frame_bottom = frame_bottom[:, frame_bottom.shape[1] // 2:] 

        # Stitch frame
        frame_bottom, _, _ = stitch_image.stitch_images(left_frame=left_frame_bottom, right_frame=right_frame_bottom, value=value, angle=angle, new_frame_size=new_frame_size_bottom, correction=correction_bottom, homography_matrix=homography_matrix_bottom)

        # Blend frame
        frame_bottom = blending.blend_image(mat=frame_bottom, intersection=params.BOTTOM["intersection"], intensity=3)
        
        # Show frame
        # utils.show_img(frame_bottom, "BOTTOM", ratio=1.5)

        #! TOP - CENTER
        frame_top_center, _, _ = stitch_image.stitch_images(left_frame=cropped_images["center_top"], right_frame=cropped_images["top"], value = params.TOP_CENTER["value"], angle = params.TOP_CENTER["angle"],
                                                         new_frame_size=new_frame_size_top_center, correction=correction_top_center, homography_matrix=homography_matrix_top_center, 
                                                         left_shift_dx = params.TOP_CENTER["left_shift_dx"], left_shift_dy = params.TOP_CENTER["left_shift_dy"], remove_offset = params.TOP_CENTER["remove_offset"]) 

        # Show frame
        # utils.show_img(frame_top_center, "TOP_CENTER", ratio=1.5)

        #! BOTTOM - CENTER
        frame_bottom_center, _, _ = stitch_image.stitch_images(left_frame=cropped_images["center_bottom"], right_frame=cropped_images["bottom"], value = params.BOTTOM_CENTER["value"], angle = params.BOTTOM_CENTER["angle"],
                                                         new_frame_size=new_frame_size_bottom_center, correction=correction_bottom_center, homography_matrix=homography_matrix_bottom_center, 
                                                         left_shift_dx = params.BOTTOM_CENTER["left_shift_dx"], left_shift_dy = params.BOTTOM_CENTER["left_shift_dy"], remove_offset = params.BOTTOM_CENTER["remove_offset"])
        
        # Show frame
        # utils.show_img(frame_bottom_center, "BOTTOM_CENTER", ratio=1.5)

        #! CENTER - CENTER
    
    # Process each video
    for video in sorted(videos):
        assert all(isfile(video) for video in videos), f"Unable to locate {video}"

if __name__ == "__main__":

    # Setup logger
    logger.setLevel(logging.INFO)
    
    handler = wrapped_logging_handler.WrappedLoggingHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)

    #? Cut video (just once)
    videos = _cut_video()

    #? Stitch video
    #_stitch_video(videos=videos)

    #? Stitch all
    _stitch_all_videos(videos=videos)

    #? Detection
    #_motion_detection(videos=videos)

    #? Tracking

    #? Team identification

    #? Ball tracking

    # Cleanup logger
    logger.removeHandler(handler)