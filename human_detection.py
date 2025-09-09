import cv2
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import os

VIDEO_SOURCE_PATH = "/content/Throwing(Negative - minute 11.00).mp4"  # Path to your input video
OUTPUT_VIDEO_PATH = "output_human_clips.mp4" # Path for the final output video
MODEL_NAME = "yolov8n.pt"  # YOLO model.
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence to consider a detection as a person.
MERGE_GAP_SECONDS = 2.0 # Merge clips if the gap between them is less than this value (in seconds).

def detect_human_segments(video_path, model, confidence_threshold):
    """
    Analyzes the video to find time segments where humans are present.

    Args:
        video_path (str): The path to the input video file.
        model (YOLO): The loaded YOLO model object.
        confidence_threshold (float): The confidence threshold for person detection.

    Returns:
        list: A list of tuples, where each tuple is a (start_time, end_time)
              segment in seconds where humans were detected.
    """
    print("Step 1: Analyzing video to find human presence...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    human_present_frames = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection on the frame
        results = model(frame, verbose=False)

        # Check if a 'person' (class 0) is detected with sufficient confidence
        human_detected = False
        for result in results:
            for box in result.boxes:
                # The 'person' class in the COCO dataset is 0
                if int(box.cls[0]) == 0 and box.conf[0] >= confidence_threshold:
                    human_detected = True
                    break
            if human_detected:
                break
        
        if human_detected:
            human_present_frames.append(frame_number)

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"  Processed {frame_number} frames...")

    cap.release()
    print(f"-> Analysis complete. Found humans in {len(human_present_frames)} frames.")

    if not human_present_frames:
        return []

    # Convert frame numbers to timestamps in seconds
    human_timestamps = np.array(human_present_frames) / fps
    
    print("Step 2: Consolidating detected frames into time segments...")
    segments = []
    if not human_timestamps.any():
        return []
        
    start_time = human_timestamps[0]
    end_time = human_timestamps[0]

    for t in human_timestamps[1:]:
        if t <= end_time + MERGE_GAP_SECONDS:
            # This frame is close to the current segment, so extend the segment
            end_time = t
        else:
            # The gap is too large, so end the current segment and start a new one
            segments.append((start_time, end_time))
            start_time = t
            end_time = t
    
    # Add the last segment to the list
    segments.append((start_time, end_time))
    
    # Add a small buffer to the end time to ensure the last action is fully captured
    final_segments = [(start, end + (1/fps)) for start, end in segments]
    print(f"-> Found {len(final_segments)} distinct human-present segments.")
    
    return final_segments


def extract_and_save_clips(video_path, segments, output_path):
    """
    Extracts the specified time segments from the video and saves them as a
    single concatenated video file.

    Args:
        video_path (str): The path to the input video file.
        segments (list): A list of (start_time, end_time) tuples.
        output_path (str): The path where the output video will be saved.
    """
    if not segments:
        print("No human segments were found to extract.")
        return

    print(f"Step 3: Extracting {len(segments)} segments and creating final video...")
    try:
        original_video = VideoFileClip(video_path)
        final_clips = []
        for i, (start, end) in enumerate(segments):
            # Ensure the clip has a valid duration
            if end > start:
                print(f"  - Extracting clip {i+1}/{len(segments)}: from {start:.2f}s to {end:.2f}s")
                clip = original_video.subclip(start, end)
                final_clips.append(clip)

        if not final_clips:
            print("No valid clips could be extracted.")
            original_video.close()
            return
            
        final_video = concatenate_videoclips(final_clips)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        # Clean up resources
        for clip in final_clips:
            clip.close()
        original_video.close()
        
        print(f"\n✅ Successfully saved the final video to '{output_path}'")

    except Exception as e:
        print(f"An error occurred during video extraction: {e}")


if __name__ == "__main__":
    # --- Check for input video ---
    if not os.path.exists(VIDEO_SOURCE_PATH):
        print(f"Error: Input video '{VIDEO_SOURCE_PATH}' not found.")
        print("Please place your video in the same directory and name it 'input.mp4', or update the VIDEO_SOURCE_PATH variable.")
    else:
        # --- Load the YOLO model ---
        print(f"Loading YOLO model ('{MODEL_NAME}'). This may take a moment...")
        try:
            yolo_model = YOLO(MODEL_NAME)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Please ensure the 'ultralytics' package is installed correctly.")
        else:
            # --- Run the detection and extraction process ---
            human_segments = detect_human_segments(VIDEO_SOURCE_PATH, yolo_model, CONFIDENCE_THRESHOLD)
            extract_and_save_clips(VIDEO_SOURCE_PATH, human_segments, OUTPUT_VIDEO_PATH)