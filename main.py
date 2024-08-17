from os import listdir, mkdir
from os.path import join, exists, basename
import sys

import logging

import cv2
import numpy as np

from src import cut_video
from src import stitch_image
from src import utils
from src import params
from src import blending
from src import motion_detection

from src import wrapped_logging_handler

# Select
MOTION_DETECTION = True
MOTION_TRACKING = False
TEAM_IDENTIFICATION = False
BALL_TRACKING = False

# Motion detection type
DETECTION_TYPE = motion_detection.BACKGROUND_SUBSTRACTION

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

    original_video_folder = params.ORIGINAL_VIDEOS_FOLDER
    original_videos = [f for f in listdir(original_video_folder) if f.endswith(".mp4")]
    
    # Check if cut folder contains all the original videos (if not, take it and cut it)
    if len(cut_videos) != len(original_videos):

        for input_video in original_videos:

            if basename(input_video) in cut_videos:
                continue

            output_video = join(cut_videos_folder, input_video)
            input_video = join(original_video_folder, input_video)
            
            # Cut video
            videos = cut_video.cut(input_video=input_video, output_video=output_video, t1=30)
    else:
        videos = cut_videos
    
    return videos

def _prepare_stitching(videos: list[str], processed_videos_folder):
    value = params.VALUE
    angle = params.ANGLE

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
            t1, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, method=cv2.LMEDS)
            new_frame_size_top, correction_top, homography_matrix_top = stitching_params

            top_config = {
                "new_frame_size": new_frame_size_top,
                "correction": correction_top,
                "homography_matrix": homography_matrix_top
            }

            # Extract reference frame for top-center stitching and use the previous params for stitching
            reference_top = utils.extract_frame(video=video_top, frame_number=130)
            left_reference_top, right_reference_top = utils.split_frame(mat=reference_top, div_left=params.TOP["div_left"], div_right=params.TOP["div_right"])
            reference_top, _, _ = stitch_image.stitch_images(left_frame=left_reference_top, right_frame=right_reference_top, value=value, angle=angle, new_frame_size=new_frame_size_top, correction=correction_top, homography_matrix=homography_matrix_top)
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
            _, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, method=cv2.LMEDS)
            new_frame_size_center, correction_center, homography_matrix_center = stitching_params

            center_config = {
                "new_frame_size": new_frame_size_center,
                "correction": correction_center,
                "homography_matrix": homography_matrix_center
            }

            # Extract reference frame for center-center stitching and use the previous params for stitching
            reference_center = utils.extract_frame(video=video_center, frame_number=130)
            left_reference_center, right_reference_center = utils.split_frame(mat=reference_center, div_left=params.CENTER["div_left"], div_right=params.CENTER["div_right"])
            reference_center, _, _ = stitch_image.stitch_images(left_frame=left_reference_center, right_frame=right_reference_center, value=value, angle=angle, new_frame_size=new_frame_size_center, correction=correction_center, homography_matrix=homography_matrix_center)
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
            frame, _, stitching_params = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, value=value, angle=angle, method=cv2.LMEDS)
            new_frame_size_bottom, correction_bottom, homography_matrix_bottom = stitching_params

            bottom_config = {
                "new_frame_size": new_frame_size_bottom,
                "correction": correction_bottom,
                "homography_matrix": homography_matrix_bottom
            }
            
            # Extract reference frame for bottom-center stitching and use the previous params for stitching
            reference_bottom = utils.extract_frame(video=video_bottom, frame_number=130)
            left_reference_bottom, right_reference_bottom = utils.split_frame(mat=reference_bottom, div_left=params.BOTTOM["div_left"], div_right=params.BOTTOM["div_right"])
            reference_bottom, _, _ = stitch_image.stitch_images(left_frame=left_reference_bottom, right_frame=right_reference_bottom, value=value, angle=angle, new_frame_size=new_frame_size_bottom, correction=correction_bottom, homography_matrix=homography_matrix_bottom)
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

    #* DEBUG
    # _, _, _ = stitch_image.stitch_images(
    #     left_frame=left_frame, 
    #     right_frame=right_frame, 
    #     value=params.FINAL["value"], 
    #     angle=params.FINAL["angle"],
    #     method=cv2.RANSAC, 
    #     new_frame_size=new_frame_size_final, 
    #     correction=correction_final, 
    #     homography_matrix=homography_matrix_final, 
    #     left_shift_dx=params.FINAL["left_shift_dx"],
    #     left_shift_dy=params.FINAL["left_shift_dy"], 
    #     remove_offset=params.FINAL["remove_offset"]
    # ) 

    assert len(set([int(video_top.get(cv2.CAP_PROP_FPS)), int(video_center.get(cv2.CAP_PROP_FPS)), int(video_bottom.get(cv2.CAP_PROP_FPS))])), "Input videos have different frame rates"
    
    # Saving all the frames in a list and then saving them is memory-consuming (lots of GBs of RAM)
    # Therefore, this solution is better
    # Create output video
    output_video = cv2.VideoWriter(
        filename=join(processed_videos_folder, "final.mp4"), # Specify output file
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"), # Specify video type
        fps=int(video_top.get(cv2.CAP_PROP_FPS)), # Same fps of the original video
        frameSize=(1425, 2358) # Specify shape (width, height)
    )

    max_frames = min([int(video.get(cv2.CAP_PROP_FRAME_COUNT)) for video in [video_top, video_center, video_bottom]]) if params.FRAMES_DEMO is None else params.FRAMES_DEMO

    return (video_top, video_center, video_bottom), (top_config, center_config, bottom_config, top_center_config, bottom_center_config, final_config), output_video, max_frames

def _stitching(videos: tuple, configs: tuple):
    top_config, center_config, bottom_config, top_center_config, bottom_center_config, final_config = configs
    video_top, video_center, video_bottom = videos

    if type(videos[0]) == cv2.VideoCapture:

        success_top, frame_top = video_top.read()
        success_center, frame_center = video_center.read()
        success_bottom, frame_bottom = video_bottom.read()

        if success_top + success_center + success_bottom != 3:
            exit()

    elif type(videos[0]) == np.ndarray:
        frame_top = video_top
        frame_center = video_center
        frame_bottom = video_bottom

    #! TOP -> Stitch top
    left_frame_top, right_frame_top = utils.split_frame(mat=frame_top, div_left=params.TOP["div_left"], div_right=params.TOP["div_right"])

    # Stitch frame
    frame_top, _, _ = stitch_image.stitch_images(
        left_frame=left_frame_top, 
        right_frame=right_frame_top, 
        value=params.VALUE, 
        angle=params.ANGLE, 
        new_frame_size=top_config["new_frame_size"], 
        correction=top_config["correction"], 
        homography_matrix=top_config["homography_matrix"]
    )

    # Blend frame
    frame_top = blending.blend_image(mat=frame_top, intersection=params.TOP["intersection"], intensity=3)

    #! CENTER -> Stitch center
    left_frame_center, right_frame_center = utils.split_frame(mat=frame_center, div_left=params.CENTER["div_left"], div_right=params.CENTER["div_right"])

    # Stitch frame
    frame_center, _, _ = stitch_image.stitch_images(
        left_frame=left_frame_center, 
        right_frame=right_frame_center, 
        value=params.VALUE, 
        angle=params.ANGLE,
        new_frame_size=center_config["new_frame_size"], 
        correction=center_config["correction"], 
        homography_matrix=center_config["homography_matrix"]
    )

    # Blend frame
    frame_center = blending.blend_image(mat=frame_center, intersection=params.CENTER["intersection"], intensity=3)

    #! BOTTOM -> Stitch bottom
    left_frame_bottom, right_frame_bottom = utils.split_frame(mat=frame_bottom, div_left=params.BOTTOM["div_left"], div_right=params.BOTTOM["div_right"])

    # Stitch frame
    frame_bottom, _, _ = stitch_image.stitch_images(
        left_frame=left_frame_bottom, 
        right_frame=right_frame_bottom, 
        value=params.VALUE, 
        angle=params.ANGLE, 
        new_frame_size=bottom_config["new_frame_size"], 
        correction=bottom_config["correction"], 
        homography_matrix=bottom_config["homography_matrix"]
    )

    # Blend frame
    frame_bottom = blending.blend_image(mat=frame_bottom, intersection=params.BOTTOM["intersection"], intensity=3)

    # Generate final frame

    # Rotate and crop the images
    bottom = utils.crop_image(cv2.rotate(frame_bottom, cv2.ROTATE_90_COUNTERCLOCKWISE))
    top = utils.crop_image(cv2.rotate(frame_top, cv2.ROTATE_90_COUNTERCLOCKWISE))
    center_for_top = utils.crop_image(cv2.rotate(frame_center, cv2.ROTATE_90_CLOCKWISE))
    center_for_bottom = utils.crop_image(cv2.rotate(frame_center, cv2.ROTATE_90_COUNTERCLOCKWISE))

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

    #TODO try to improve stitching
    # ##############################################################################################################
    # top_stitched_image = cv2.warpPerspective(top, homography_matrix_top_center, (new_frame_size_top_center[1], new_frame_size_top_center[0]))
    # top_stitched_image = utils.crop_image(cv2.rotate(top_stitched_image, cv2.ROTATE_180))

    # center_top_stitched_image = np.zeros(top_stitched_image.shape, dtype=np.uint8)

    # region_x_start = 0 + correction_top_center[0]
    # region_x_end = correction_top_center[0] + center_for_top.shape[1] + 0 - 0
    # region_y_start = correction_top_center[1] + 0
    # region_y_end = correction_top_center[1] + center_for_top.shape[0] + 0

    # region_x_end = min(region_x_end, center_top_stitched_image.shape[1])
    # region_y_end = min(region_y_end, center_top_stitched_image.shape[0])

    # center_top_stitched_image[region_y_start:region_y_end, region_x_start:region_x_end] = center_for_top[:region_y_end-region_y_start, :region_x_end-region_x_start]
    # ##############################################################################################################

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

    #TODO try to improve stitching
    # ##############################################################################################################
    # bottom_stitched_image = cv2.warpPerspective(bottom, homography_matrix_bottom_center, (new_frame_size_bottom_center[1], new_frame_size_bottom_center[0]))

    # center_bottom_stitched_image = np.zeros(bottom_stitched_image.shape, dtype=np.uint8)

    # region_x_start = 0 + correction_bottom_center[0]
    # region_x_end = correction_bottom_center[0] + center_for_bottom.shape[1] + 0 - 0
    # region_y_start = correction_bottom_center[1] + 0
    # region_y_end = correction_bottom_center[1] + center_for_bottom.shape[0] + 0

    # region_x_end = min(region_x_end, center_top_stitched_image.shape[1])
    # region_y_end = min(region_y_end, center_top_stitched_image.shape[0])

    # center_bottom_stitched_image[region_y_start:region_y_end, region_x_start:region_x_end] = center_for_bottom[:region_y_end-region_y_start, :region_x_end-region_x_start]
    # ##############################################################################################################

    #! FINAL -> Stitch all the views
    left_frame = utils.crop_image(cv2.rotate(frame_top_center, cv2.ROTATE_180))
    right_frame = utils.crop_image(frame_bottom_center)

    frame_final, _, _ = stitch_image.stitch_images(
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

    #TODO try to improve stitching
    # ##############################################################################################################
    # bottom_stitched_image = utils.crop_image(bottom_stitched_image)
    # bottom_stitched_image = cv2.warpPerspective(bottom_stitched_image, homography_matrix_final, (new_frame_size_final[1], new_frame_size_final[0]))

    # center_bottom_stitched_image = utils.crop_image(center_bottom_stitched_image)
    # center_bottom_stitched_image = cv2.warpPerspective(center_bottom_stitched_image, homography_matrix_final, (new_frame_size_final[1], new_frame_size_final[0]))
    # ##############################################################################################################

    # Crop
    frame_final = frame_final[300:-300, 150:-150]

    return frame_final

def _motion_detection(frame_final, frame_background, configs):
    # For the first frame of the video
    if frame_background is None:
        if DETECTION_TYPE == motion_detection.FRAME_SUBSTRACTION:
            frame_background = frame_final        
        elif DETECTION_TYPE == motion_detection.BACKGROUND_SUBSTRACTION:
            frame_top = utils.extract_frame("videos/cut/top.mp4", 5390)
            frame_center = utils.extract_frame("videos/cut/center.mp4", 5390)
            frame_bottom = utils.extract_frame("videos/cut/bottom.mp4", 5390)

            frames = (frame_top, frame_center, frame_bottom)

            frame_background = _stitching(videos=frames, configs=configs)
        elif DETECTION_TYPE == motion_detection.ADAPTIVE_BACKGROUND_SUBSTRACTION:
            #TODO pass both background frame and last frame
            frame_background = frame_final 
            
    # Apply motion detection
    motion_frame, _, bw = motion_detection.detection(frame=frame_final, detection_type=DETECTION_TYPE, background=frame_background)

    # Update the background frame for frame subtraction method
    if DETECTION_TYPE == motion_detection.FRAME_SUBSTRACTION:
            frame_background = frame_final
    
    return motion_frame, frame_background, bw

def processing(videos: list[str], live: bool = True) -> None:
    
    # Create workspace
    processed_videos_folder = params.FINAL_STITCHED_VIDEOS_FOLDER
    frame_background = None
    
    if not exists(processed_videos_folder):
        mkdir(processed_videos_folder)

    videos, configs, output_video, max_frames = _prepare_stitching(videos, processed_videos_folder)
    video_top, video_center, video_bottom = videos

    while True:
        #! Stitching
        frame_final = _stitching(videos, configs)
        
        #! Motion detection
        if MOTION_DETECTION:
            frame_final, frame_background, bw = _motion_detection(frame_final=frame_final, frame_background=frame_background, configs=configs)
            winname = "Motion detection" 
        
        #! Motion tracking
        elif MOTION_TRACKING:
            # motion_tracking = ...
            winname = "Motion tracking"
        
        #! Team identification
        elif TEAM_IDENTIFICATION:
            winname = "Team identification"
        
        #! Ball tracking
        elif BALL_TRACKING:
            winname = "Ball tracking"
            
        else:
            winname = "Stitching"

        # Save processed frame
        output_video.write(frame_final)

        # Show frame
        if live:

            cv2.imshow(winname=winname, mat=cv2.resize(bw, (bw.shape[1]//2, bw.shape[0]//2)))
            cv2.imshow(winname="", mat=cv2.resize(frame_final, (frame_final.shape[1]//2, frame_final.shape[0]//2)))
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        
        # Interrupt processing in case of max_frames < total number of frames in the video
        if video_top.get(cv2.CAP_PROP_POS_FRAMES) == max_frames:
            break

    # Cleanup
    if live:
        cv2.destroyAllWindows()

    output_video.release()
    video_top.release()
    video_center.release()
    video_bottom.release()

if __name__ == "__main__":

    # Setup logger
    logger.setLevel(logging.INFO)
    
    handler = wrapped_logging_handler.WrappedLoggingHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)

    #? Cut video (just once)
    videos = _cut_video()

    #? Process videos
    processing(videos=videos)

    # Cleanup logger
    logger.removeHandler(handler)