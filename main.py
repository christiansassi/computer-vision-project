from os import listdir, mkdir
from os.path import join, exists, basename

import cv2
import numpy as np

import signal
import sys
import inspect

from src import cut_video
from src import stitch_image
from src import utils
from src import params
from src import blending
from src import motion_detection
from src import motion_tracking

from src import wrapped_logging_handler

# Select
MOTION_DETECTION = True
MOTION_TRACKING = True
TEAM_IDENTIFICATION = False
BALL_TRACKING = False

OUTPUT_VIDEO = None

def cleanup(signum, frame):

    global OUTPUT_VIDEO

    if OUTPUT_VIDEO is not None:
        OUTPUT_VIDEO.release()

    sys.exit(0)



def _cut_video(videos: list[str]) -> list[str]:

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
            cut_videos.append(cut_video.cut(input_video=input_video, output_video=output_video, t1=30))
    
    return cut_videos

def _stitching(frame_top: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, frame_center: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, frame_bottom: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, videos: list[str] = [], calculate_params: bool = False) -> np.ndarray:

    function = eval(inspect.stack()[0][3])

    # If calculate_params is set to false but no params have been cached, calculate them
    try:
        function.params

    except:
        calculate_params = True

    # If specified, calculate stitching params    
    if calculate_params:
        
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

def _motion_detection(frame: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, detection_type: int, time_window: int = 1, background: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat = None, alpha: float = None, reset: bool = False) -> tuple[np.ndarray, list[tuple]]:

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

        return motion_detection.frame_substraction(mat=frame, time_window=time_window, reset=reset)

    elif detection_type == motion_detection.BACKGROUND_SUBSTRACTION:

        # Apply background substraction
        #* PROS
        #* [+] Good since the background doesn't change too much (for this purpose)
        #* [+] Keeps detecting objects even if they stop moving

        #! CONS
        #! [-] None (for this purpose)

        assert background is not None, "Invalid background"

        return motion_detection.background_substraction(mat=frame, background=background)

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

        return motion_detection.adaptive_background_substraction(mat=frame, background=background, alpha=alpha, reset=reset)

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

        return motion_detection.gaussian_average(mat=frame, background=background, alpha=alpha, reset=reset)



def process_videos(videos: list[str], live: bool = True) -> None:
    
    global OUTPUT_VIDEO

    # Create workspace
    processed_videos_folder = params.PROCESSED_VIDEOS_FOLDER

    if not exists(processed_videos_folder):
        mkdir(processed_videos_folder)

    # Open videos
    for video in videos:

        video_capture = cv2.VideoCapture(video)
        assert video_capture.isOpened(), "An error occours while opening the video"
        
        if "top" in video:
            video_top = video_capture
        
        elif "center" in video:
            video_center = video_capture
        
        elif "bottom" in video:
            video_bottom = video_capture
        
        else:
            raise Exception(f"Unknown video {video}")

    output_video = None
    
    background = None

    # Process videos
    while True:

        success_top, frame_top = video_top.read()
        success_center, frame_center = video_center.read()
        success_bottom, frame_bottom = video_bottom.read()

        if sum([success_top, success_center, success_bottom]) != 3:
            break

        #! Stitching
        stitched_frame = _stitching(frame_top=frame_top, frame_center=frame_center, frame_bottom=frame_bottom, videos=videos)
        
        processed_frame = stitched_frame

        #! Motion detection
        if MOTION_DETECTION:
            
            # Extract background (only once)
            if background is None:
                extracted_frame_top = utils.extract_frame(video=video_top, frame_number=params.BACKGROUND_FRAME)
                extracted_frame_center = utils.extract_frame(video=video_center, frame_number=params.BACKGROUND_FRAME)
                extracted_frame_bottom = utils.extract_frame(video=video_bottom, frame_number=params.BACKGROUND_FRAME)
                background = _stitching(frame_top=extracted_frame_top, frame_center=extracted_frame_center, frame_bottom=extracted_frame_bottom, videos=videos)

            motion_detection_frame, motion_detection_bounding_boxes = _motion_detection(frame=stitched_frame, detection_type=motion_detection.BACKGROUND_SUBSTRACTION, background=background)

            processed_frame = motion_detection_frame

        #! Motion tracking
        if MOTION_TRACKING and MOTION_DETECTION:
            motion_tracking_frame, motion_tracking_results = motion_tracking.particle_filtering(mat=processed_frame, bounding_boxes=motion_detection_bounding_boxes)

            processed_frame = motion_tracking_frame

        #! Team identification
        # if TEAM_IDENTIFICATION:
        #     pass

        #! Ball tracking
        # if BALL_TRACKING:
        #     pass

        # Show processed video
        if live:

            cv2.imshow("Processed video", processed_frame)

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

    # Setup logger
    logger = wrapped_logging_handler.get_logger()

    # List original videos (to be processed)
    videos = [join(params.ORIGINAL_VIDEOS_FOLDER, f) for f in listdir(params.ORIGINAL_VIDEOS_FOLDER) if f.endswith(".mp4")]

    #? Cut video (just once)
    videos = _cut_video(videos=videos)

    #? Process videos
    process_videos(videos=videos)