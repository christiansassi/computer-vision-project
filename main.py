import os
from os import listdir, mkdir, system, remove
from os.path import join, exists, basename, isfile
import signal
import sys
import inspect
from ultralytics import YOLO
from time import time
import statistics
import cv2
import numpy as np

# Clear screen
clear_screen = lambda: system("cls") if os.name == "nt" else system("clear")
clear_screen()

# Setup logger
from src import wrapped_logging_handler
logger = wrapped_logging_handler.get_logger()

# Custom modules
logger.info(f"Importing modules...")
from src import blending
from src import cut_video
from src import motion_detection
from src import motion_tracking
from src import team_identification
from src import params
from src import stitch_image
from src import utils
from src import ball_tracking
from src import draw_tracking_points
print("DONE")

# Select
MOTION_DETECTION = True
MOTION_TRACKING = True
TEAM_IDENTIFICATION = True
BALL_DETECTION = True
BALL_TRACKING = True

OUTPUT_VIDEO = None

def cleanup(signum, frame):

    global OUTPUT_VIDEO

    if OUTPUT_VIDEO is not None:
        
        sys.stdout.write("")
        sys.stdout.write(f"\033[B" * params.NUM_LINES)
        sys.stdout.flush()

        print("\n")

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

def __motion_detection(frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, detection_type: int, time_window: int = 1, background: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat = None, alpha: float = None, min_area: int = None, max_area: int = None, reset: bool = False) -> list[tuple]:

    assert detection_type in [1,2,3, 4], "Invalid motion detection type"

    if detection_type == motion_detection.FRAME_SUBTRACTION:

        # Apply frame subtraction
        #* PROS
        #* [+] None (for this purpose)

        #! CONS
        #! [-] Stops detecting an object if it stops moving
        #! [-] A larger window can avoid the previous problem but would negatively impact detection quality

        # Apply background subtraction
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

        return motion_detection.frame_subtraction(**args)

    elif detection_type == motion_detection.BACKGROUND_SUBTRACTION:

        # Apply background subtraction
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

        return motion_detection.background_subtraction(**args)

    elif detection_type == motion_detection.ADAPTIVE_BACKGROUND_SUBTRACTION:

        # Apply adaptive subtraction
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

        return motion_detection.adaptive_background_subtraction(**args)

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

def __ball_detection(frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, model: YOLO) -> dict | None:

    # Resize the frame 
    original_frame = frame.copy()
    resized_frame = cv2.resize(original_frame, (800, 800))

    # Detect objects
    results = model(resized_frame, verbose=False)

    # Dimensions of the original and resized frames
    h_orig, w_orig = original_frame.shape[:2]
    h_resized, w_resized = resized_frame.shape[:2]

    # Scale factors
    scale_x = w_orig / w_resized
    scale_y = h_orig / h_resized

    # Extract bounding boxes
    best_ball = {}

    for result in results:
        for box in result.boxes:
            
            # Get confidence
            confidence = box.conf[0]

            # Extract class label
            class_id = int(box.cls[0])
            label = params.YOLO_CLASS_MAP.get(class_id, params.YOLO_CLASS.UNKNOWN)

            # Draw bounding box based on a threshold
            if confidence > params.YOLO_CONFIDENCE and label == params.YOLO_CLASS.BALL and best_ball.get("confidence", -1) < confidence:

                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Convert actual coordinates into original image coordinates
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x1 - x2)
                h = abs(y1 - y2)

                text = f"{label.value}: {confidence:.2f}"

                best_ball = {
                    "confidence": confidence,
                    "bounding_box": (x, y, w, h),
                    "text": text
                }

    return best_ball if len(best_ball) else None

def process_videos(videos: list[str], live: bool = True) -> None:
    
    global OUTPUT_VIDEO
    
    # Load the model for ball and player detection
    model = YOLO(params.YOLO_PATH, verbose=False)

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
    skip_to = 1190
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
    logger.info(f"\033[32mPress Ctrl+C to exit\033[0m\n\n")

    # Performance evaluation
    elapsed = time()
    fps = 0
    times = []

    # Ball tracking
    ball_points = []
    team1_points = []
    team2_points = []

    while True:

        success_top, frame_top = video_top.read()
        success_center, frame_center = video_center.read()
        success_bottom, frame_bottom = video_bottom.read()

        if sum([success_top, success_center, success_bottom]) != 3:
            break

        #! Stitching
        stitching_time = time()
        stitched_frame = __stitching(frame_top=frame_top, frame_center=frame_center, frame_bottom=frame_bottom, videos=videos)
        stitching_time = time() - stitching_time

        processed_frame = stitched_frame

        #! Motion detection
        if MOTION_DETECTION:
            motion_detection_time = time()
            motion_detection_bounding_boxes = __motion_detection(frame=stitched_frame, detection_type=motion_detection.BACKGROUND_SUBTRACTION, background=background, min_area=4000)
            motion_detection_time = time() - motion_detection_time
        else:
            motion_detection_bounding_boxes = []
            motion_detection_time = None
        
        #! Motion tracking
        if MOTION_TRACKING and len(motion_detection_bounding_boxes):
            motion_tracking_time = time()
            motion_tracking_results = motion_tracking.particle_filtering(mat=processed_frame, bounding_boxes=motion_detection_bounding_boxes)
            motion_tracking_time = time() - motion_tracking_time

        else:
            motion_tracking_results = {}
            motion_tracking_time = None

        #! Team identification
        if TEAM_IDENTIFICATION and len(motion_detection_bounding_boxes):
            team1, team2 = team_identification.identify_teams(bounding_boxes=motion_detection_bounding_boxes)

            motion_tracking_team1 = {}
            motion_tracking_team2 = {}

            for bounding_box in motion_tracking_results:
                
                if bounding_box in team1:
                    motion_tracking_team1[bounding_box] = motion_tracking_results[bounding_box]
                
                elif bounding_box in team2:
                    motion_tracking_team2[bounding_box] = motion_tracking_results[bounding_box]
        else:
            team1, team2 = [], []
            motion_tracking_team1, motion_tracking_team2 = {}, {}

        #! Ball detection
        if BALL_DETECTION:
            ball_detection_time = time()
            ball = __ball_detection(frame=processed_frame, model=model)
            ball_detection_time = time() - ball_detection_time
        else:
            ball = None
            ball_detection_time = None
    
        #! Ball tracking
        if BALL_TRACKING and BALL_DETECTION and ball is not None:
            ball_tracking_time = time()
            ball_tracking_results = ball_tracking.particle_filtering(mat=processed_frame, bounding_box=ball["bounding_box"])
            ball_tracking_time = time() - ball_tracking_time
        else:
            ball_tracking_results = {}
            ball_tracking_time = None

        other_time = time()

        # Draw
        team1_players = []
        team2_players = []
        team_players = []
        ball_points = []

        if TEAM_IDENTIFICATION:
            for x, y, w, h in team1:
                cv2.putText(processed_frame, params.TEAM1_LABEL, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, params.TEAM1_COLOR, 2)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), params.TEAM1_COLOR, 2)
            
            for x, y, w, h in team2:
                cv2.putText(processed_frame, params.TEAM2_LABEL, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, params.TEAM2_COLOR, 2)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), params.TEAM2_COLOR, 2)

            for obj in list(motion_tracking_team1.values()):
                origin = obj["origin"]
                estimated = obj["estimated"]
                team1_players.append(obj["points"])

                cv2.arrowedLine(processed_frame, origin, estimated, params.TEAM1_COLOR, 4, tipLength=0.25)

            for obj in list(motion_tracking_team2.values()):
                origin = obj["origin"]
                estimated = obj["estimated"]
                team2_players.append(obj["points"])

                cv2.arrowedLine(processed_frame, origin, estimated,params.TEAM2_COLOR, 4, tipLength=0.25)

        elif MOTION_DETECTION:
            for x, y, w, h in motion_detection_bounding_boxes:
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), params.TEAM_DEFAULT_COLOR, 2)

            for obj in list(motion_tracking_results.values()):
                origin = obj["origin"]
                estimated = obj["estimated"]
                team_players.append(obj["points"])

                cv2.arrowedLine(processed_frame, origin, estimated, params.TEAM_DEFAULT_COLOR, 4, tipLength=0.25)

        if ball is not None:

            x, y, w, h = ball["bounding_box"]
            text = ball["text"]

            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), params.BALL_COLOR, 2)
            cv2.putText(processed_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, params.BALL_COLOR, 2)

            for obj in list(ball_tracking_results.values()):
                origin = obj["origin"]
                estimated = obj["estimated"]
                ball_points = obj["points"]

                cv2.arrowedLine(processed_frame, origin, estimated, params.BALL_COLOR, 4, tipLength=0.25)

        draw_tracking_points.draw_points(team1_players=team1_players, team2_players=team2_players, team_players=team_players, ball_points=ball_points)
        
        # Show processed video
        if live:

            cv2.imshow("Processed video", cv2.resize(processed_frame, (processed_frame.shape[1] // 2, processed_frame.shape[0] // 2)))
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        
        # Save processed video
        if output_video is None:

            if isfile(params.PROCESSED_VIDEO):
                remove(params.PROCESSED_VIDEO)

            output_video = cv2.VideoWriter(
                filename=params.PROCESSED_VIDEO, # Specify output file
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"), # Specify video type
                fps=18, #int(min(video_top.get(cv2.CAP_PROP_FPS), video_center.get(cv2.CAP_PROP_FPS), video_bottom.get(cv2.CAP_PROP_FPS))), # Same fps of the original video
                frameSize=processed_frame.shape[:2][::-1] # Specify shape (width, height)
            )

            OUTPUT_VIDEO = output_video

        if time() - elapsed > 1:
            elapsed = time()
            fps = 1
        else:
            fps = fps + 1

        output_video.write(processed_frame)

        # Calculate total time
        total_time = sum([t for t in [stitching_time, motion_detection_time, motion_tracking_time, ball_detection_time, ball_tracking_time] if t is not None])
        other_time = time() - other_time

        total_time = total_time + other_time

        # Calculate AVG. time
        times.append(total_time)

        if len(times) >= 10:
            times.pop(0)

        avg_total_time = statistics.mean(times)

        # Print performances
        info_log1 = (
            f"{' '*100}\n"
            f"{' '*100}\n"
            f"{' '*100}\n"
            f"{' '*100}\n"
            f"{' '*100}\n"
            f"{' '*100}\n"
            f"{' '*100}\n"
            f"{' '*100}\n"
            f"{' '*100}"
        )

        info_log2 = (
            f"Processing             {int(video_top.get(cv2.CAP_PROP_POS_FRAMES)) + 1} / {total_frames_number}  \n\n"

            f"Stitching time:        {round(stitching_time, 2)}  \n"
            f"Motion detection:      {round(motion_detection_time, 2) if motion_detection_time is not None else '-'}  \n"
            f"Motion tracking:       {round(motion_tracking_time, 2) if motion_tracking_time is not None else '-'}  \n"
            f"Ball detection:        {round(ball_detection_time, 2) if ball_detection_time is not None else '-'}  \n"
            f"Ball tracking:         {round(ball_tracking_time, 2) if ball_tracking_time is not None else '-'}  \n"

            f"Avg. total time:       {round(avg_total_time, 2)}  \n"
            f"FPS:                   {round(fps, 2)}  "
        )

        sys.stdout.write(info_log1)
        sys.stdout.write(f"\033[F" * params.NUM_LINES)
        sys.stdout.flush()

        sys.stdout.write(info_log2)
        sys.stdout.write(f"\033[F" * params.NUM_LINES)
        sys.stdout.flush()

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
    process_videos(videos=videos, live=False)