"""
Copyright 2024 Abram Jackson
See LICENSE
"""

import os
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    clips_array,
    ColorClip,
)
from collections import defaultdict
import argparse

# Configuration
AUDIO_FILE = "audio/input_audio.wav"          # Path to your main audio file
RTTM_FILE = "diarization/diarization.rttm"    # Path to your RTTM file
SOURCE_IMAGES_DIR = "source_images/"          # Directory containing static images
OUTPUT_VIDEOS_DIR = "output_videos/"          # Directory containing video chunks
FINAL_VIDEO = "final_combined_video.mp4"      # Output final video file

SPEAKER_LEFT = "SPEAKER_01"    # Speaker on the left
SPEAKER_RIGHT = "SPEAKER_00"   # Speaker on the right

NUM_POSES = 4                  # Number of pose images per speaker
TRANSITION_DURATION = 0.5      # Duration of fade transitions in seconds
GAP_THRESHOLD = 1.2            # Maximum gap (in seconds) to merge segments

def parse_rttm(rttm_path, gap_threshold):
    """
    Parse the RTTM file and merge consecutive segments of the same speaker
    if the gap between them is less than or equal to the specified threshold.

    Args:
        rttm_path (str): Path to the RTTM file.
        gap_threshold (float): Maximum allowed gap (in seconds) to merge segments.

    Returns:
        List of dictionaries with keys: 'speaker', 'start', 'end'
    """
    segments = []
    with open(rttm_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # Skip malformed lines
            speaker_label = parts[7]  # e.g., SPEAKER_01
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            segments.append({
                'speaker': speaker_label,
                'start': start_time,
                'end': end_time
            })

    # Sort segments by start time
    segments.sort(key=lambda x: x['start'])

    # Merge segments
    merged_segments = []
    if not segments:
        return merged_segments

    current = segments[0]
    for next_seg in segments[1:]:
        if (next_seg['speaker'] == current['speaker'] and
            (next_seg['start'] - current['end']) <= gap_threshold):
            # Merge segments
            current['end'] = next_seg['end']
        else:
            merged_segments.append(current)
            current = next_seg
    merged_segments.append(current)  # Append the last segment

    return merged_segments

def get_total_duration(segments, audio_path):
    """
    Determine the total duration of the audio.

    Args:
        segments (list): List of segment dictionaries.
        audio_path (str): Path to the main audio file.

    Returns:
        float: Total duration in seconds.
    """
    if segments:
        max_end = max(segment['end'] for segment in segments)
    else:
        # If no segments, get duration from audio file
        audio_clip = AudioFileClip(audio_path)
        max_end = audio_clip.duration
        audio_clip.close()
    return max_end

def get_next_pose_image(speaker, pose_counters):
    """
    Get the next pose image for a speaker.

    Args:
        speaker (str): Speaker label, e.g., 'SPEAKER_01'.
        pose_counters (dict): Dictionary tracking pose counts per speaker.

    Returns:
        str: Path to the selected pose image.
    """
    pose_index = pose_counters[speaker] % NUM_POSES
    pose_counters[speaker] += 1  # Increment only when a static image is inserted
    image_filename = f"{speaker}_pose_{pose_index}.png"
    image_path = os.path.join(SOURCE_IMAGES_DIR, image_filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Static image not found: {image_path}")
    return image_path

def load_video_chunk(speaker, chunk_index):
    """
    Load a video chunk for a speaker.

    Args:
        speaker (str): Speaker label, e.g., 'SPEAKER_01'.
        chunk_index (int): Index of the chunk for this speaker.

    Returns:
        VideoFileClip or None: Loaded video clip or None if not found.
    """
    chunk_filename = f"chunk_{chunk_index:02d}_speaker_{speaker.split('_')[-1]}.mp4"
    chunk_path = os.path.join(OUTPUT_VIDEOS_DIR, chunk_filename)
    if not os.path.exists(chunk_path):
        return None  # Return None if chunk does not exist
    clip = VideoFileClip(chunk_path)
    # Resize to 512x512 if not already
    if clip.size != (512, 512):
        clip = clip.resize(newsize=(512, 512))
    # Remove audio to prevent multiple audio tracks
    clip = clip.without_audio()
    return clip

def create_static_image_clip(image_path, duration, fade_in=False, fade_duration=TRANSITION_DURATION):
    """
    Create a video clip from a static image.

    Args:
        image_path (str): Path to the image file.
        duration (float): Duration of the clip in seconds.
        fade_in (bool): Whether to apply a fade-in effect.
        fade_duration (float): Duration of the fade-in in seconds.

    Returns:
        ImageClip: Created image clip.
    """
    clip = ImageClip(image_path).set_duration(duration)
    # Resize to 512x512 if not already
    if clip.size != (512, 512):
        clip = clip.resize(newsize=(512, 512))
    if fade_in:
        clip = clip.fadein(fade_duration)
    return clip

def build_speaker_clips(speaker, segments, total_duration, pose_counters):
    """
    Build a list of video clips for a speaker, aligned with the global timeline.

    Args:
        speaker (str): Speaker label, e.g., 'SPEAKER_01'.
        segments (list): List of segment dictionaries for the speaker.
        total_duration (float): Total duration of the audio.
        pose_counters (dict): Dictionary tracking pose counts per speaker.

    Returns:
        list: List of tuples (clip, start_time).
    """
    clips = []
    current_time = 0.0
    chunk_index = -1  # Initialize chunk index for the speaker

    for segment in segments:
        chunk_index += 1
        speak_start = segment['start']
        speak_end = segment['end']

        # Handle silent period before the current speaking segment
        if speak_start > current_time:
            silent_duration = speak_start - current_time
            if silent_duration > 0:
                # Insert static image for silent period
                try:
                    static_image_path = get_next_pose_image(speaker, pose_counters)
                    static_clip = create_static_image_clip(
                        static_image_path,
                        silent_duration,
                        fade_in=True,
                        fade_duration=TRANSITION_DURATION
                    )
                except FileNotFoundError:
                    # Use black screen if static image not found
                    static_clip = ColorClip(size=(512, 512), color=(0, 0, 0)).set_duration(silent_duration)
                clips.append((static_clip, current_time))
                print(f"Info, adding static image at {current_time} until {current_time + silent_duration}")
                current_time += silent_duration

        # Insert speaking video chunk
        speaking_clip = load_video_chunk(speaker, chunk_index)
        if speaking_clip:
            clips.append((speaking_clip, speak_start))
            print(f"Info, adding video chunk (speaker {speaker} chunk {chunk_index}) at {current_time} until {speak_end}")
            current_time = speak_end
        else:
            # It's the other speaker's turn
            # TODO manage this better, it won't handle some scenarios
            chunk_index += 1
            speaking_clip = load_video_chunk(speaker, chunk_index)
            clips.append((speaking_clip, speak_start))
            print(f"Info, adding video chunk (speaker {speaker} chunk {chunk_index}) at {current_time} until {speak_end}")
            current_time = speak_end

    # Handle silent period after the last speaking segment
    if current_time < total_duration:
        silent_duration = total_duration - current_time
        if silent_duration > 0:
            try:
                static_image_path = get_next_pose_image(speaker, pose_counters)
                static_clip = create_static_image_clip(
                    static_image_path,
                    silent_duration,
                    fade_in=True,
                    fade_duration=TRANSITION_DURATION
                )
            except FileNotFoundError:
                # Use black screen if static image not found
                static_clip = ColorClip(size=(512, 512), color=(0, 0, 0)).set_duration(silent_duration)
            clips.append((static_clip, current_time))
            print(f"Info, adding static image at {current_time} until end of video")
            current_time += silent_duration

    return clips

def main():
    parser = argparse.ArgumentParser(description="Combine lip-synced video chunks into a synchronized side-by-side video.")
    args = parser.parse_args()

    # Parse and merge RTTM file
    merged_segments = parse_rttm(RTTM_FILE, GAP_THRESHOLD)
    print(f"Total merged segments: {len(merged_segments)}")

    # Determine total duration
    total_duration = get_total_duration(merged_segments, AUDIO_FILE)
    print(f"Total duration of audio: {total_duration:.2f} seconds")

    # Organize segments per speaker
    speaker_segments = defaultdict(list)
    for segment in merged_segments:
        speaker_segments[segment['speaker']].append(segment)

    # Initialize pose counters for each speaker
    pose_counters = defaultdict(int)

    # Build video clips for each speaker
    if SPEAKER_LEFT not in speaker_segments:
        print(f"Warning: {SPEAKER_LEFT} has no segments.")
    if SPEAKER_RIGHT not in speaker_segments:
        print(f"Warning: {SPEAKER_RIGHT} has no segments.")

    left_clips = build_speaker_clips(
        speaker=SPEAKER_LEFT,
        segments=speaker_segments[SPEAKER_LEFT],
        total_duration=total_duration,
        pose_counters=pose_counters
    )

    right_clips = build_speaker_clips(
        speaker=SPEAKER_RIGHT,
        segments=speaker_segments[SPEAKER_RIGHT],
        total_duration=total_duration,
        pose_counters=pose_counters
    )

    # Create video tracks for both speakers using CompositeVideoClip without duration
    speaker_left_video = CompositeVideoClip(
        [clip.set_start(start_time) for clip, start_time in left_clips],
        size=(512, 512)
    )
    speaker_right_video = CompositeVideoClip(
        [clip.set_start(start_time) for clip, start_time in right_clips],
        size=(512, 512)
    )

    # Combine the two speaker clips side by side
    combined_clip = clips_array([[speaker_left_video, speaker_right_video]]).set_duration(total_duration)

    # Load the main audio
    try:
        main_audio = AudioFileClip(AUDIO_FILE)
    except Exception as e:
        print(f"Error loading main audio file: {e}")
        return

    # Set the audio to the combined video
    final_video = combined_clip.set_audio(main_audio)

    # Write the final video to file
    print("Exporting the final video. This may take a while...")
    final_video.write_videofile(
        FINAL_VIDEO,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        threads=4,
        preset="medium"
    )

    # Close all clips
    speaker_left_video.close()
    speaker_right_video.close()
    combined_clip.close()
    main_audio.close()
    final_video.close()
    print(f"Final video saved as {FINAL_VIDEO}")

if __name__ == "__main__":
    main()
