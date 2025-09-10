from moviepy.editor import VideoFileClip
import moviepy
import math
import os

print(moviepy.__version__)

def split_video_into_clips(video_path, output_dir, clip_duration):
    video = VideoFileClip(video_path)
    total_duration = math.floor(video.duration) 

    os.makedirs(output_dir, exist_ok=True)

    for start_time in range(0, total_duration, clip_duration):
        end_time = min(start_time + clip_duration, total_duration)
        clip = video.subclip(start_time, end_time)
        output_file = os.path.join(output_dir, f"clip_{start_time}_{end_time}.mp4")
        clip.write_videofile(output_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        print(f"Saved: {output_file}")

    video.close()

split_video_into_clips("human_clips/output_human_clips_throwing_positive.mp4", "onlyHumansFrame_throwing_positive_10s_output_clips", clip_duration=10)

# python video_clipping.py