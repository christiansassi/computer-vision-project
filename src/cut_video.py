from os import remove
from os.path import isfile

# Use moviepy since it is faster than opencv
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

import proglog
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def cut(input_video: str, output_video: str, t1: int = None, t2: int = None, clear: bool = True, log: bool = True) -> str:

    if not log:
        logging.disable(logging.CRITICAL)

        logging.info(f"Processing: {input_video}")

    # Disable moviepy command logs
    #! This line invalidate the default_bar_logger for the entire script!
    proglog.default_bar_logger = lambda *args, **kwargs: lambda *inner_args, **inner_kwargs: None

    # Verify that video exists
    assert isfile(input_video), f"Video '{input_video}' doesn't exist"

    # Delete output video if it alrady exists and if clear flag is enabled
    if clear and isfile(output_video):
        remove(output_video)

    # Open video
    clip = VideoFileClip(filename=input_video) 

    # Close it, otherwise we can have handling problems
    clip.reader.close()

    # Extract subclip
    t1 = t1 if t1 is not None else 0
    t2 = t2 if t2 is not None else clip.duration

    ffmpeg_extract_subclip(filename=input_video, t1=t1, t2=t2, targetname=output_video)

    logging.info(f"Done: {output_video}")

    return output_video