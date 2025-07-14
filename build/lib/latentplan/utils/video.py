import os
import numpy as np
import skvideo.io

def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

# def save_video(filename, video_frames, fps=60, video_format='mp4'):
#     assert fps == int(fps), fps
#     _make_dir(filename)
#
#     skvideo.io.vwrite(
#         filename,
#         video_frames,
#         inputdict={
#             '-r': str(int(fps)),
#         },
#         outputdict={
#             '-f': video_format,
#             '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
#         }
#     )

def save_video(filename, video_frames, fps=60, video_format='mp4', resolution=(1920, 1080)):
    assert fps == int(fps), fps
    _make_dir(filename)

    # Ensure video frames are in the correct resolution
    # Assuming video_frames is a numpy array of shape (num_frames, height, width, channels)
    # If the resolution of the frames is different from the desired resolution, you need to resize the frames
    # Resize frames to the desired resolution
    import cv2
    resized_frames = [cv2.resize(frame, resolution) for frame in video_frames]

    skvideo.io.vwrite(
        filename,
        resized_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for macOS compatibility
            '-vf': f'scale={resolution[0]}:{resolution[1]}'  # Scale filter to ensure resolution
        }
    )

def save_videos(filename, *video_frames, **kwargs):
    ## video_frame : [ N x H x W x C ]
    video_frames = np.concatenate(video_frames, axis=2)
    save_video(filename, video_frames, **kwargs)
