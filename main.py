import os
from os import listdir, mkdir, system
from os.path import join, exists, basename
import signal
import sys
import inspect
from inference import get_model
from ultralytics import YOLO

import cv2
import numpy as np
import math
from collections import defaultdict

# Clear screen
if os.name == "nt":
    system("cls")
else:
    system("clear")

# Setup logger
from src import wrapped_logging_handler
logger = wrapped_logging_handler.get_logger()

# Custom modules
logger.info(f"Importing modules...")
from src import blending
from src import cut_video
from src import motion_detection
from src import motion_tracking
from src import params
from src import stitch_image
from src import utils
print("DONE")

# Select
MOTION_DETECTION = False
MOTION_TRACKING = False
TEAM_IDENTIFICATION = False
BALL_TRACKING = False
BALL_TRACKING_YOLO = True

OUTPUT_VIDEO = None

def cleanup(signum, frame):

    global OUTPUT_VIDEO

    if OUTPUT_VIDEO is not None:
        print("")
        logger.info(f"Saving video...\n")
        OUTPUT_VIDEO.release()
        logger.info(f"Video saved to '{params.PROCESSED_VIDEO}'\n")

    sys.exit(0)

def __cut_video(videos: list[str]) -> list[str]:

    # Check if cut videos already exist. If not create the workspace
    cut_videos_folder = params.CUT_VIDEOS_FOLDER

    if exists(cut_videos_folder):
        cut_videos = [join(cut_videos_folder, f) for f in listdir(cut_videos_folder) if f.endswith(".mp4")]

    else:
        mkdir(cut_videos_folder)
        cut_videos = []
    
    # Check if cut folder contains all the original videos (if not, take it and cut it)
    if len(cut_videos) != len(videos):

        for input_video in videos:

            if basename(input_video) in cut_videos:
                continue
            
            output_video = join(cut_videos_folder, basename(input_video))
            
            # Cut video
            logger.info(f"Cutting '{input_video}'...\n")
            cut_videos.append(cut_video.cut(input_video=input_video, output_video=output_video, t1=30))
            logger.info(f"Video saved to '{output_video}'\n")
    
    return cut_videos

def __stitching(frame_top: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, frame_center: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, frame_bottom: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, videos: list[str] = [], calculate_params: bool = False) -> np.ndarray:

    function = eval(inspect.stack()[0][3])

    # If calculate_params is set to false but no params have been cached, calculate them
    try:
        function.params

    except:
        calculate_params = True

    # If specified, calculate stitching params    
    if calculate_params:
        
        logger.info(f"Calculating stitching parameters...")

        assert len(videos), "Videos list empty"

        # Params of each video
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
        
        # Calculate the stitching params for each view by taking into account the reference frames
        # In this way we do not have to re-calculate them when processing the entire video
        for video in videos:

            video_capture = cv2.VideoCapture(video)
            assert video_capture.isOpened(), "An error occours while opening the video"

            if "top" in video:
                video_top = video_capture

                # Extract reference frame
                frame = utils.extract_frame(video=video_top, frame_number=params.TOP["frame_number"])

                # Calculate stitching params but here we only process the shared parts to facilitate features extraction
                left_frame, right_frame = utils.split_frame(mat=frame, div_left=params.TOP["div_left"], div_right=params.TOP["div_right"])
                left_frame, right_frame = utils.black_box_on_image(left_frame=left_frame, right_frame=right_frame, left_width=params.TOP["left_width"], right_width=params.TOP["right_width"])
                _, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=params.VALUE, angle=params.ANGLE, method=cv2.LMEDS)
                new_frame_size_top, correction_top, homography_matrix_top = stitching_params

                top_config = {
                    "new_frame_size": new_frame_size_top,
                    "correction": correction_top,
                    "homography_matrix": homography_matrix_top
                }

                # Extract reference frame for top-center stitching and use the previous params for stitching
                reference_top = utils.extract_frame(video=video_top, frame_number=130)
                left_reference_top, right_reference_top = utils.split_frame(mat=reference_top, div_left=params.TOP["div_left"], div_right=params.TOP["div_right"])
                reference_top, _, _ = stitch_image.stitch_images(left_frame=left_reference_top, right_frame=right_reference_top, value=params.VALUE, angle=params.ANGLE, new_frame_size=new_frame_size_top, correction=correction_top, homography_matrix=homography_matrix_top)
                reference_top = blending.blend_image(mat=reference_top, intersection=params.TOP["intersection"], intensity=3)

                # Apply jpg compression to the image. 
                # During tests we noticed that this procedure helps finding better features
                reference_top = utils.jpg_compression(mat=reference_top)

                # Rotate and crop the image
                top = utils.crop_image(cv2.rotate(reference_top, cv2.ROTATE_90_COUNTERCLOCKWISE))

            elif "center" in video:
                video_center = video_capture

                # Extract reference frame
                frame = utils.extract_frame(video=video_center, frame_number=params.CENTER["frame_number"])

                # Calculate stitching params but here we only process the shared parts to facilitate features extraction
                left_frame, right_frame = utils.split_frame(mat=frame, div_left=params.CENTER["div_left"], div_right=params.CENTER["div_right"])
                left_frame, right_frame = utils.black_box_on_image(left_frame=left_frame, right_frame=right_frame, left_width=params.CENTER["left_width"], right_width=params.CENTER["right_width"])
                _, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=params.VALUE, angle=params.ANGLE, method=cv2.LMEDS)
                new_frame_size_center, correction_center, homography_matrix_center = stitching_params

                center_config = {
                    "new_frame_size": new_frame_size_center,
                    "correction": correction_center,
                    "homography_matrix": homography_matrix_center
                }

                # Extract reference frame for center-center stitching and use the previous params for stitching
                reference_center = utils.extract_frame(video=video_center, frame_number=130)
                left_reference_center, right_reference_center = utils.split_frame(mat=reference_center, div_left=params.CENTER["div_left"], div_right=params.CENTER["div_right"])
                reference_center, _, _ = stitch_image.stitch_images(left_frame=left_reference_center, right_frame=right_reference_center, value=params.VALUE, angle=params.ANGLE, new_frame_size=new_frame_size_center, correction=correction_center, homography_matrix=homography_matrix_center)
                reference_center = blending.blend_image(mat=reference_center, intersection=params.CENTER["intersection"], intensity=3)
                
                # Apply jpg compression to the image. 
                # During tests we noticed that this procedure helps finding better features
                reference_center = utils.jpg_compression(mat=reference_center)

                # Rotate and crop the image
                center_for_top = utils.crop_image(cv2.rotate(reference_center, cv2.ROTATE_90_CLOCKWISE))
                center_for_bottom = utils.crop_image(cv2.rotate(reference_center, cv2.ROTATE_90_COUNTERCLOCKWISE))

            elif "bottom" in video:
                video_bottom = video_capture

                # Extract reference frame
                frame = utils.extract_frame(video=video_bottom, frame_number=params.BOTTOM["frame_number"])

                # Calculate stitching params but here we only process the shared parts to facilitate features extraction
                left_frame, right_frame = utils.split_frame(mat=frame, div_left=params.BOTTOM["div_left"], div_right=params.BOTTOM["div_right"])
                left_frame, right_frame = utils.black_box_on_image(left_frame=left_frame, right_frame=right_frame, left_width=params.BOTTOM["left_width"], right_width=params.BOTTOM["right_width"])
                frame, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=params.VALUE, angle=params.ANGLE, method=cv2.LMEDS)
                new_frame_size_bottom, correction_bottom, homography_matrix_bottom = stitching_params

                bottom_config = {
                    "new_frame_size": new_frame_size_bottom,
                    "correction": correction_bottom,
                    "homography_matrix": homography_matrix_bottom
                }
                
                # Extract reference frame for bottom-center stitching and use the previous params for stitching
                reference_bottom = utils.extract_frame(video=video_bottom, frame_number=130)
                left_reference_bottom, right_reference_bottom = utils.split_frame(mat=reference_bottom, div_left=params.BOTTOM["div_left"], div_right=params.BOTTOM["div_right"])
                reference_bottom, _, _ = stitch_image.stitch_images(left_frame=left_reference_bottom, right_frame=right_reference_bottom, value=params.VALUE, angle=params.ANGLE, new_frame_size=new_frame_size_bottom, correction=correction_bottom, homography_matrix=homography_matrix_bottom)
                reference_bottom = blending.blend_image(mat=reference_bottom, intersection=params.BOTTOM["intersection"], intensity=3)
                
                # Apply jpg compression to the image. 
                # During tests we noticed that this procedure helps finding better features
                reference_bottom = utils.jpg_compression(mat=reference_bottom)

                # Rotate and crop the image
                bottom = utils.crop_image(cv2.rotate(reference_bottom, cv2.ROTATE_90_COUNTERCLOCKWISE))

            else:
                raise Exception("Unknwon video")

        # Now calculate the stitching params for the final view (the one that combines top, center and bottom views)

        #! TOP_CENTER -> Stitch top and center views

        #TODO better define this
        left_frame, right_frame = utils.bb(left_frame=center_for_top, right_frame=top, left_min=params.TOP_CENTER["left_min"], left_max=center_for_top.shape[1], right_min=params.TOP_CENTER["right_min"], right_max=params.TOP_CENTER["right_max"])

        # Calculate stitching params
        _, _, stitching_params = stitch_image.stitch_images(
            left_frame=left_frame, 
            right_frame=right_frame, 
            value=params.TOP_CENTER["value"], 
            angle=params.TOP_CENTER["angle"], 
            method=cv2.RANSAC, 
            user_left_kp=params.TOP_CENTER["left_frame_kp"], 
            user_right_kp=params.TOP_CENTER["right_frame_kp"]
        )

        new_frame_size_top_center, correction_top_center, homography_matrix_top_center = stitching_params

        top_center_config = {
            "new_frame_size": new_frame_size_top_center,
            "correction": correction_top_center,
            "homography_matrix": homography_matrix_top_center
        }

        # Extract reference frame for the final stitching
        reference_top_center, _, _ = stitch_image.stitch_images(
            left_frame=center_for_top, 
            right_frame=top, 
            value=params.TOP_CENTER["value"], 
            angle=params.TOP_CENTER["angle"],
            method=cv2.RANSAC, 
            new_frame_size=new_frame_size_top_center, 
            correction=correction_top_center, 
            homography_matrix=homography_matrix_top_center, 
            left_shift_dx=params.TOP_CENTER["left_shift_dx"], 
            left_shift_dy= params.TOP_CENTER["left_shift_dy"], 
            remove_offset=params.TOP_CENTER["remove_offset"]
        ) 

        #! BOTTOM_CENTER -> Stitch bottom and center views 

        # Calculate stitching params and extract reference frame for the final stitching
        reference_bottom_center, _, stitching_params = stitch_image.stitch_images(
            left_frame=center_for_bottom, 
            right_frame=bottom, 
            value=params.BOTTOM_CENTER["value"], 
            angle=params.BOTTOM_CENTER["angle"], 
            method=cv2.RANSAC,
            left_shift_dx=params.BOTTOM_CENTER["left_shift_dx"], 
            left_shift_dy= params.BOTTOM_CENTER["left_shift_dy"], 
            remove_offset=params.BOTTOM_CENTER["remove_offset"]
        )

        new_frame_size_bottom_center, correction_bottom_center, homography_matrix_bottom_center = stitching_params
        
        bottom_center_config = {
            "new_frame_size": new_frame_size_bottom_center,
            "correction": correction_bottom_center,
            "homography_matrix": homography_matrix_bottom_center
        }

        #! FINAL -> Stitch all the views

        # Apply jpg compression to the images. 
        # During tests we noticed that this procedure helps finding better features
        reference_top_center = utils.jpg_compression(mat=reference_top_center)
        reference_bottom_center = utils.jpg_compression(mat=reference_bottom_center)

        # Crop and rotate images
        reference_top_center = utils.crop_image(cv2.rotate(reference_top_center.copy(), cv2.ROTATE_180))
        reference_bottom_center = utils.crop_image(reference_bottom_center.copy())

        left_frame = reference_top_center.copy()
        right_frame = reference_bottom_center.copy()

        #TODO better define this
        left_frame, right_frame = utils.bb(left_frame=left_frame, right_frame=right_frame, left_min=params.FINAL["left_min"], left_max=params.FINAL["left_max"], right_min=params.FINAL["right_min"], right_max=params.FINAL["right_max"])

        _, _, stitching_params = stitch_image.stitch_images(
            left_frame=left_frame, 
            right_frame=right_frame, 
            value=params.FINAL["value"], 
            angle=params.FINAL["angle"],
            method=cv2.RANSAC, 
            user_left_kp=params.FINAL["left_frame_kp"], 
            user_right_kp=params.FINAL["right_frame_kp"],
            left_shift_dx=params.FINAL["left_shift_dx"], 
            left_shift_dy= params.FINAL["left_shift_dy"], 
            remove_offset=params.FINAL["remove_offset"]
        )
        
        new_frame_size_final, correction_final, homography_matrix_final = stitching_params

        final_config = {
            "new_frame_size": new_frame_size_final,
            "correction": correction_final,
            "homography_matrix": homography_matrix_final
        }

        function.params = top_config, center_config, bottom_config, top_center_config, bottom_center_config, final_config
        function.videos = video_top, video_center, video_bottom

        print("DONE")

    # Calculate stitched image
    _frame_top = frame_top.copy()
    _frame_center = frame_center.copy()
    _frame_bottom = frame_bottom.copy()

    top_config, center_config, bottom_config, top_center_config, bottom_center_config, final_config = function.params

    #! TOP -> Stitch top
    left_frame_top, right_frame_top = utils.split_frame(mat=_frame_top, div_left=params.TOP["div_left"], div_right=params.TOP["div_right"])

    # Stitch frame
    _frame_top, _, _ = stitch_image.stitch_images(
        left_frame=left_frame_top, 
        right_frame=right_frame_top, 
        value=params.VALUE, 
        angle=params.ANGLE, 
        new_frame_size=top_config["new_frame_size"], 
        correction=top_config["correction"], 
        homography_matrix=top_config["homography_matrix"]
    )

    # Blend frame
    _frame_top = blending.blend_image(mat=_frame_top, intersection=params.TOP["intersection"], intensity=3)

    #! CENTER -> Stitch center
    left_frame_center, right_frame_center = utils.split_frame(mat=_frame_center, div_left=params.CENTER["div_left"], div_right=params.CENTER["div_right"])

    # Stitch frame
    _frame_center, _, _ = stitch_image.stitch_images(
        left_frame=left_frame_center, 
        right_frame=right_frame_center, 
        value=params.VALUE, 
        angle=params.ANGLE,
        new_frame_size=center_config["new_frame_size"], 
        correction=center_config["correction"], 
        homography_matrix=center_config["homography_matrix"]
    )

    # Blend frame
    _frame_center = blending.blend_image(mat=_frame_center, intersection=params.CENTER["intersection"], intensity=3)

    #! BOTTOM -> Stitch bottom
    left_frame_bottom, right_frame_bottom = utils.split_frame(mat=_frame_bottom, div_left=params.BOTTOM["div_left"], div_right=params.BOTTOM["div_right"])

    # Stitch frame
    _frame_bottom, _, _ = stitch_image.stitch_images(
        left_frame=left_frame_bottom, 
        right_frame=right_frame_bottom, 
        value=params.VALUE, 
        angle=params.ANGLE, 
        new_frame_size=bottom_config["new_frame_size"], 
        correction=bottom_config["correction"], 
        homography_matrix=bottom_config["homography_matrix"]
    )

    # Blend frame
    _frame_bottom = blending.blend_image(mat=_frame_bottom, intersection=params.BOTTOM["intersection"], intensity=3)

    # Generate final frame

    # Rotate and crop the images
    bottom = utils.crop_image(cv2.rotate(_frame_bottom, cv2.ROTATE_90_COUNTERCLOCKWISE))
    top = utils.crop_image(cv2.rotate(_frame_top, cv2.ROTATE_90_COUNTERCLOCKWISE))
    center_for_top = utils.crop_image(cv2.rotate(_frame_center, cv2.ROTATE_90_CLOCKWISE))
    center_for_bottom = utils.crop_image(cv2.rotate(_frame_center, cv2.ROTATE_90_COUNTERCLOCKWISE))

    #! TOP_CENTER -> Stitch top and center views
    frame_top_center, _, _ = stitch_image.stitch_images(
        left_frame=center_for_top, 
        right_frame=top, 
        value=params.TOP_CENTER["value"], 
        angle=params.TOP_CENTER["angle"],
        new_frame_size=top_center_config["new_frame_size"], 
        correction=top_center_config["correction"], 
        homography_matrix=top_center_config["homography_matrix"],
        left_shift_dx=params.TOP_CENTER["left_shift_dx"], 
        left_shift_dy=params.TOP_CENTER["left_shift_dy"], 
        remove_offset=params.TOP_CENTER["remove_offset"]
    ) 

    #! BOTTOM_CENTER -> Stitch bottom and center views
    frame_bottom_center, _, _ = stitch_image.stitch_images(
        left_frame=center_for_bottom, 
        right_frame=bottom, 
        value=params.BOTTOM_CENTER["value"], 
        angle=params.BOTTOM_CENTER["angle"],
        new_frame_size=bottom_center_config["new_frame_size"], 
        correction=bottom_center_config["correction"], 
        homography_matrix=bottom_center_config["homography_matrix"],
        left_shift_dx=params.BOTTOM_CENTER["left_shift_dx"], 
        left_shift_dy=params.BOTTOM_CENTER["left_shift_dy"], 
        remove_offset=params.BOTTOM_CENTER["remove_offset"]
    )

    #! FINAL -> Stitch all the views
    left_frame = utils.crop_image(cv2.rotate(frame_top_center, cv2.ROTATE_180))
    right_frame = utils.crop_image(frame_bottom_center)

    stitched_frame, _, _ = stitch_image.stitch_images(
        left_frame=left_frame, 
        right_frame=right_frame, 
        value=params.FINAL["value"], 
        angle=params.FINAL["angle"],
        new_frame_size=final_config["new_frame_size"], 
        correction=final_config["correction"], 
        homography_matrix=final_config["homography_matrix"],
        left_shift_dx=params.FINAL["left_shift_dx"], 
        left_shift_dy=params.FINAL["left_shift_dy"], 
        remove_offset=params.FINAL["remove_offset"]
    )

    # Crop
    return stitched_frame[300:-300, 150:-150]

def __motion_detection(frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, detection_type: int, time_window: int = 1, background: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat = None, alpha: float = None, min_area: int = None, max_area: int = None, reset: bool = False) -> tuple[np.ndarray, list[tuple]]:

    assert detection_type in [1,2,3, 4], "Invalid motion detection type"

    if detection_type == motion_detection.FRAME_SUBSTRACTION:

        # Apply frame substraction
        #* PROS
        #* [+] None (for this purpose)

        #! CONS
        #! [-] Stops detecting an object if it stops moving
        #! [-] A larger window can avoid the previous problem but would negatively impact detection quality

        # Apply background substraction
        #* PROS
        #* [+] Good since the background doesn't change too much (for this purpose)
        #* [+] Keeps detecting objects even if they stop moving

        #! CONS
        #! [-] None (for this purpose)

        assert isinstance(time_window, int), "Invalid time window"
        assert time_window > 0, "Invalid time window"

        args = {
            "mat": frame,
            "time_window": time_window,
            "reset": reset
        }

        if min_area is not None:
            assert isinstance(min_area, int) and min_area > 0, "Invalid minimum area"
            args["min_area"] = min_area
        
        if max_area is not None:
            assert isinstance(max_area, int) and max_area > 0, "Invalid maximum area"
            args["max_area"] = max_area

        return motion_detection.frame_substraction(**args)

    elif detection_type == motion_detection.BACKGROUND_SUBSTRACTION:

        # Apply background substraction
        #* PROS
        #* [+] Good since the background doesn't change too much (for this purpose)
        #* [+] Keeps detecting objects even if they stop moving

        #! CONS
        #! [-] None (for this purpose)

        assert background is not None, "Invalid background"

        args = {
            "mat": frame,
            "background": background,
        }

        if min_area is not None:
            assert isinstance(min_area, int) and min_area > 0, "Invalid minimum area"
            args["min_area"] = min_area

        if max_area is not None:
            assert isinstance(max_area, int) and max_area > 0, "Invalid maximum area"
            args["max_area"] = max_area

        return motion_detection.background_substraction(**args)

    elif detection_type == motion_detection.ADAPTIVE_BACKGROUND_SUBSTRACTION:

        # Apply adaptive substraction
        #* PROS
        #* [+] Good for this purpose since the background doesn't change too much
        #* [+] Compared to normal background subtraction, it adapts to small background changes

        #! CONS
        #! [-] A large alpha value causes the algorithm to stop detecting objects that have stopped moving
        #! [-] Since we are forced to use a small alpha value, this algorithm becomes similar to normal background subtraction

        assert background is not None, "Invalid background"

        assert isinstance(alpha, (int, float)), "Invalid alpha"
        assert alpha >= 0 and alpha <= 1, "Alpha must be a number in the interval [0, 1]"

        args = {
            "mat": frame,
            "background": background,
            "alpha": alpha,
            "reset": reset,
        }

        if min_area is not None:
            assert isinstance(min_area, int) and min_area > 0, "Invalid minimum area"
            args["min_area"] = min_area

        if max_area is not None:
            assert isinstance(max_area, int) and max_area > 0, "Invalid maximum area"
            args["max_area"] = max_area

        return motion_detection.adaptive_background_substraction(**args)

    elif detection_type == motion_detection.GAUSSIAN_AVERAGE:

        # Apply gaussian average motion detection
        #* PROS
        #* [+] Good for this purpose since the background doesn't change too much
        #* [+] It adapts to small background changes

        #! CONS
        #! [-] Because the environment remains relatively stable (with minimal likelihood of scene changes such as illumination shifts), we opt for a smaller alpha value. 
        #!     However, using a smaller alpha makes this approach resemble background subtraction techniques in its behavior

        assert background is not None, "Invalid background"

        assert isinstance(alpha, (int, float)), "Invalid alpha"
        assert alpha >= 0 and alpha <= 1, "Alpha must be a number in the interval [0, 1]"

        args = {
            "mat": frame,
            "background": background,
            "alpha": alpha,
            "reset": reset,
        }

        if min_area is not None:
            assert isinstance(min_area, int) and min_area > 0, "Invalid minimum area"
            args["min_area"] = min_area

        if max_area is not None:
            assert isinstance(max_area, int) and max_area > 0, "Invalid maximum area"
            args["max_area"] = max_area

        return motion_detection.gaussian_average(**args)

def process_videos(videos: list[str], live: bool = True) -> None:
    
    global OUTPUT_VIDEO
    
    # Load the model for ball and player detection
    model = YOLO(os.path.join("models", "best_v11_1300_noaug.pt"), verbose=False)
    track_history = defaultdict(lambda: [])

    # Create workspace
    logger.info(f"Creating workspace...")

    processed_videos_folder = params.PROCESSED_VIDEOS_FOLDER

    if not exists(processed_videos_folder):
        mkdir(processed_videos_folder)

    print("DONE")

    # Open videos

    for video in videos:
        
        logger.info(f"Opening '{video}'...")

        video_capture = cv2.VideoCapture(video)
        assert video_capture.isOpened(), "An error occours while opening the video"
        
        print("DONE")

        if "top" in video:
            video_top = video_capture
        
        elif "center" in video:
            video_center = video_capture
        
        elif "bottom" in video:
            video_bottom = video_capture
        
        else:
            raise Exception(f"Unknown video {video}")

    total_frames_number = int(video_top.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fast forward the videos
    skip_to = 1150
    video_top.set(cv2.CAP_PROP_POS_FRAMES, skip_to)
    video_center.set(cv2.CAP_PROP_POS_FRAMES, skip_to)
    video_bottom.set(cv2.CAP_PROP_POS_FRAMES, skip_to)
    
    # Extract background
    extracted_frame_top = utils.extract_frame(video=video_top, frame_number=params.BACKGROUND_FRAME)
    extracted_frame_center = utils.extract_frame(video=video_center, frame_number=params.BACKGROUND_FRAME)
    extracted_frame_bottom = utils.extract_frame(video=video_bottom, frame_number=params.BACKGROUND_FRAME)
    background = __stitching(frame_top=extracted_frame_top, frame_center=extracted_frame_center, frame_bottom=extracted_frame_bottom, videos=videos)

    output_video = None

    # Process videos
    logger.info(f"\033[32mPress Ctrl+C to exit\033[0m\n")

    while True:

        success_top, frame_top = video_top.read()
        success_center, frame_center = video_center.read()
        success_bottom, frame_bottom = video_bottom.read()

        if sum([success_top, success_center, success_bottom]) != 3:
            break
        
        logger.info(f"Processing {int(video_top.get(cv2.CAP_PROP_POS_FRAMES))} / {total_frames_number}\r")

        #! Stitching
        stitched_frame = __stitching(frame_top=frame_top, frame_center=frame_center, frame_bottom=frame_bottom, videos=videos)
        
        processed_frame = stitched_frame

        #! Motion detection
        if MOTION_DETECTION:
            motion_detection_frame, motion_detection_bounding_boxes = __motion_detection(frame=stitched_frame, detection_type=motion_detection.BACKGROUND_SUBSTRACTION, background=background, min_area=4000)
            #processed_frame = motion_detection_frame

        #! Motion tracking
        if MOTION_TRACKING and MOTION_DETECTION:
            motion_tracking_frame, motion_tracking_results = motion_tracking.particle_filtering(mat=processed_frame, bounding_boxes=motion_detection_bounding_boxes)
            processed_frame = motion_tracking_frame

        #! Team identification
        # if TEAM_IDENTIFICATION:
        #     pass

        #! Ball tracking
        def evaluate_circle_similarity(contour):
            # Calcola l'area del contorno
            contour_area = cv2.contourArea(contour)

            # Trova il cerchio minimo che racchiude il contorno
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # Calcola l'area del cerchio minimo
            circle_area = np.pi * (radius ** 2)

            # Rapporto area contorno/area cerchio
            if circle_area > 0:
                ratio = contour_area / circle_area
            else:
                ratio = 0

            return ratio, (int(x), int(y)), int(radius)
        
        if BALL_TRACKING:
            # processed_frame = cv2.imread("frame.png")
            # utils.show_img(processed_frame, ratio=1.5)

            # Converti l'immagine in formato HSV
            hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)

            # Definisci i range di colore per il giallo e il blu (modificati per la scena)
            lower_yellow = np.array([29, 58, 142], dtype=np.uint8)
            upper_yellow = np.array([34, 122, 232], dtype=np.uint8)

            lower_blue = np.array([120, 61, 17], dtype=np.uint8)
            upper_blue = np.array([130, 237, 92], dtype=np.uint8)

            # Yellow mask
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            yellow_contours = []

            for contour in contours:
                area = cv2.contourArea(contour)

                if not 0 < area < 200:
                    cv2.drawContours(mask_yellow, [contour], contourIdx=-1, color=(0, 0, 0), thickness=cv2.FILLED)
                    continue
                
                M = cv2.moments(contour)

                # Calculate the centroid
                if M["m00"] != 0:  # Avoid division by zero
                    cx = int(M["m10"] / M["m00"])  # x-coordinate of the centroid
                    cy = int(M["m01"] / M["m00"])  # y-coordinate of the centroid
                    yellow_contours.append([contour, (cx, cy)])

            kernel = np.ones((9, 9), np.uint8)
            mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)
            # utils.show_img(mask_yellow, ratio=1.5)
            
            # Blue mask
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                
                M = cv2.moments(contour)

                # Calculate the centroid
                if M["m00"] != 0:  # Avoid division by zero
                    cx = int(M["m10"] / M["m00"])  # x-coordinate of the centroid
                    cy = int(M["m01"] / M["m00"])  # y-coordinate of the centroid
                    center = (cx, cy)
                else:
                    cv2.drawContours(mask_blue, [contour], contourIdx=-1, color=(0, 0, 0), thickness=cv2.FILLED)
                    continue

                skip = True

                for _, yellow_center in yellow_contours:

                    if math.dist(yellow_center, center) < 50:
                        skip = False
                        break
                
                if skip:
                    cv2.drawContours(mask_blue, [contour], contourIdx=-1, color=(0, 0, 0), thickness=cv2.FILLED)

            kernel = np.ones((9, 9), np.uint8)
            mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)
            # utils.show_img(mask_blue, ratio=1.5)

            # for _, center in yellow_contours:
            #     cv2.circle(mask_yellow, center, 20, (255, 255, 255), thickness=cv2.FILLED)

            # Combina le maschere
            mask = cv2.bitwise_or(mask_yellow, mask_blue)

            # utils.show_img(mask, ratio=1.5)

            # Applica un leggero blur per ridurre il rumore
            # blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Applica dilation
            kernel = np.ones((9, 9), np.uint8)
            blurred_mask = cv2.dilate(mask, kernel, iterations=1)
            blurred_mask = cv2.erode(blurred_mask, kernel, iterations=1)

            # utils.show_img(blurred_mask, ratio=1.5)
 
            # Trova i contorni degli oggetti nella maschera pulita
            contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # utils.show_img(cv2.drawContours(processed_frame.copy(), contours, -1, (0, 255, 0), 2), ratio=1.5)

            # Per ogni contorno, valuta quanto somiglia a un cerchio
            best_circle_similarity = 0
            best_circle_info = None

            for contour in contours:
                area = cv2.contourArea(contour)
                
                if 200 < area < 2000:  # Filtra contorni piccoli
                    similarity_ratio, center, radius = evaluate_circle_similarity(contour)
                    
                    # Se il rapporto è più vicino a 1, somiglia di più a un cerchio
                    if similarity_ratio > best_circle_similarity:
                        best_circle_similarity = similarity_ratio
                        best_circle_info = (center, radius, area)

            # Se abbiamo trovato un contorno che somiglia a un cerchio, disegna il cerchio
            processed_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            if best_circle_info:
                center, radius, area = best_circle_info
                cv2.circle(processed_frame, center, radius, (0, 255, 0), 2)  # Verde con spessore 2

            #motion_detection_frame, motion_detection_bounding_boxes = __motion_detection(frame=stitched_frame, detection_type=motion_detection.BACKGROUND_SUBSTRACTION, background=background, min_area=1000, max_area=2500)
            #processed_frame = motion_detection_frame

            pass
        
        if BALL_TRACKING_YOLO:

            # Ridimensiona l'immagine a 640x640 per il modello
            original_frame = processed_frame.copy()
            resized_frame = cv2.resize(original_frame, (800, 800))

            # Effettua la detection e tracking sull'immagine ridimensionata
            results = model.track(resized_frame, verbose=False, tracker="bytetrack.yaml")

            # Ottieni le dimensioni originali
            h_orig, w_orig = original_frame.shape[:2]
            h_resized, w_resized = resized_frame.shape[:2]

            # Fattori di scala per riproiettare le bounding box sull'immagine originale
            scale_x = w_orig / w_resized
            scale_y = h_orig / h_resized

            # Ottieni i bounding box e riproiettali sull'immagine originale
            for result in results:
                for box, id in zip(result.boxes, result.boxes.id):
                    # Coordinate della bounding box sulla versione ridimensionata
                    x, y, w, h = map(int, box.xywh[0])
                    
                    # Class and confidence
                    class_id = int(box.cls[0])
                    confidence = box.conf[0]

                    # Tracking ID
                    track_id = box.id[0] if box.id is not None else None  # Ottieni l'ID del tracking se presente

                    # Mappa l'id della classe su un nome
                    class_map = {0: "ball", 1: "player"} 
                    label = class_map.get(class_id, "unknown")

                    if label == "ball":
                        track = track_history[int(box.id)]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)
                        
                        scaled_points = np.copy(track)
                        scaled_points[:, 0] = scaled_points[:, 0] * scale_x  # Rescale x-coordinates
                        scaled_points[:, 1] = scaled_points[:, 1] * scale_y  # Rescale y-coordinates

                        # Convert to the required format for polylines
                        scaled_points = np.hstack(scaled_points).astype(np.int32).reshape((-1, 1, 2))

                        # Draw the polylines on the processed frame with the rescaled points
                        cv2.polylines(processed_frame, [scaled_points], isClosed=False, color=(255, 0, 0), thickness=5)

                    if label == "player":
                        color = (0, 0, 255)  # Rosso
                    elif label == "ball":
                        color = (0, 255, 0)  # Verde
                    else:
                        color = (255, 255, 255)  # Bianco

                    # Ripristina le coordinate sulla dimensione originale
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)

                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    # Disegna la bounding box solo se la confidenza è maggiore di una soglia
                    if confidence > 0.5:  # Soglia di confidenza per filtrare i risultati
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 3)

                        # Aggiungi la label, la confidenza, e l'ID del tracking
                        text = f"{label}: {confidence:.2f}"
                        if track_id is not None:
                            text += f" ID: {track_id}"

                        cv2.putText(processed_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show processed video
        if live:
            
            # cv2.imwrite(f"training_dataset/frame_{int(video_top.get(cv2.CAP_PROP_POS_FRAMES))}.png", processed_frame)
            cv2.imshow("Processed video", cv2.resize(processed_frame, (processed_frame.shape[1] // 2, processed_frame.shape[0] // 2)))
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        
        # Save processed video
        if output_video is None:

            output_video = cv2.VideoWriter(
                filename=params.PROCESSED_VIDEO, # Specify output file
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"), # Specify video type
                fps=int(min(video_top.get(cv2.CAP_PROP_FPS), video_center.get(cv2.CAP_PROP_FPS), video_bottom.get(cv2.CAP_PROP_FPS))), # Same fps of the original video
                frameSize=processed_frame.shape[:2][::-1] # Specify shape (width, height)
            )

            OUTPUT_VIDEO = output_video

        output_video.write(processed_frame)

    # Cleanup
    cv2.destroyAllWindows()

    video_top.release()
    video_center.release()
    video_bottom.release()

    output_video.release()

if __name__ == "__main__":

    # Handle system signals
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # List original videos (to be processed)
    videos = [join(params.ORIGINAL_VIDEOS_FOLDER, f) for f in listdir(params.ORIGINAL_VIDEOS_FOLDER) if f.endswith(".mp4")]

    #? Cut video (just once)
    videos = __cut_video(videos=videos)

    #? Process videos
    process_videos(videos=videos)