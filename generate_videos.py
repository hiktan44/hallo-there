"""
Copyright 2024 Abram Jackson
See LICENSE
"""

import os
import subprocess
import tempfile
from pydub import AudioSegment
import argparse
from collections import defaultdict

# Configuration
AUDIO_FILE = "audio/input_audio.wav"          # Path to your main audio file
RTTM_FILE = "diarization/diarization.rttm"    # Path to your RTTM file
SOURCE_IMAGES_DIR = "source_images/"          # Directory containing source images
INFERENCE_SCRIPT = "scripts/inference.py"             # Path to Hallo's inference script
OUTPUT_VIDEOS_DIR = "output_videos/"          # Directory to save output videos

MERGE_GAP_THRESHOLD = 1.2                      # Maximum gap (in seconds) to merge segments

# Ensure output directory exists
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

# Initialize speaker pose counters
speaker_pose_counters = defaultdict(int)

def parse_and_merge_rttm(rttm_path, gap_threshold):
    """
    Parse the RTTM file and merge consecutive segments of the same speaker
    if the gap between them is less than the specified threshold.

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

def extract_audio_chunk(audio_path, start, end):
    """
    Extract a chunk of audio from the main audio file.

    Args:
        audio_path (str): Path to the main audio file.
        start (float): Start time in seconds.
        end (float): End time in seconds.

    Returns:
        Path to the temporary audio chunk file.
    """
    audio = AudioSegment.from_file(audio_path)
    start_ms = max(start * 1000, 0)  # Ensure non-negative
    end_ms = min(end * 1000, len(audio))  # Ensure not exceeding audio length
    chunk = audio[start_ms:end_ms]

    # Create a temporary file to save the audio chunk
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    chunk.export(temp_file.name, format="wav")
    return temp_file.name

def get_source_image(speaker, pose_counters):
    """
    Determine the source image for the given speaker based on the pose counter.

    Args:
        speaker (str): Speaker label, e.g., 'SPEAKER_01'.
        pose_counters (dict): Dictionary tracking pose counts per speaker.

    Returns:
        Path to the selected source image.
    """
    pose_index = pose_counters[speaker] % 4  # Assuming 4 poses: 0-3
    pose_counters[speaker] += 1
    image_filename = f"{speaker}_pose_{pose_index}.png"
    image_path = os.path.join(SOURCE_IMAGES_DIR, image_filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source image not found: {image_path}")
    return image_path

def run_inference(source_image, driving_audio, output_video):
    """
    Call the Hallo project's inference script with the specified parameters.

    Args:
        source_image (str): Path to the source image.
        driving_audio (str): Path to the driving audio file.
        output_video (str): Path to save the output video.
    """
    command = [
        "python", INFERENCE_SCRIPT,
        "--source_image", source_image,
        "--driving_audio", driving_audio,
        "--output", output_video,
        # Add other arguments if needed, e.g., weights
        # "--pose_weight", "0.8",
        # "--face_weight", "1.0",
        # "--lip_weight", "1.2",
        # "--face_expand_ratio", "1.1"
    ]

    try:
        print(f"Running inference: {' '.join(command)}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during inference: {e}")
        raise

def generate_chunks(merged_segments, mode):
    """
    Generate video chunks based on merged segments.

    Args:
        merged_segments (list): List of merged segment dictionaries.
        mode (str): 'chunks' or 'full'.
    """
    for idx, segment in enumerate(merged_segments):
        speaker = segment['speaker']
        start = segment['start']
        end = segment['end']
        duration = end - start

        print(f"Processing chunk {idx:02d}: Speaker={speaker}, Start={start:.3f}, Duration={duration:.3f} seconds")

        # Extract audio chunk
        try:
            audio_chunk_path = extract_audio_chunk(AUDIO_FILE, start, end)
        except Exception as e:
            print(f"Failed to extract audio chunk {idx:02d}: {e}")
            continue

        # Select source image
        try:
            source_image = get_source_image(speaker, speaker_pose_counters)
        except FileNotFoundError as e:
            print(e)
            os.unlink(audio_chunk_path)  # Clean up
            continue

        # Define output video path
        speaker_id = speaker.split('_')[-1]  # Extract '01' from 'SPEAKER_01'
        output_video = os.path.join(OUTPUT_VIDEOS_DIR, f"chunk_{idx:02d}_speaker_{speaker_id}.mp4")

        # Run inference
        try:
            run_inference(source_image, audio_chunk_path, output_video)
            print(f"Generated video: {output_video}")
        except Exception as e:
            print(f"Failed to generate video for chunk {idx:02d}: {e}")
        finally:
            # Clean up temporary audio file
            os.unlink(audio_chunk_path)

def main():
    parser = argparse.ArgumentParser(description="Generate lip-synced video chunks from audio based on diarization RTTM.")
    parser.add_argument(
        "--mode",
        choices=["chunks", "full"],
        default="chunks",
        help="Mode of operation: 'chunks' to generate video only during speaking segments, 'full' to generate a complete video covering the entire audio duration."
    )
    args = parser.parse_args()

    mode = args.mode
    print(f"Running in '{mode}' mode.")

    # Parse and merge RTTM file
    merged_segments = parse_and_merge_rttm(RTTM_FILE, MERGE_GAP_THRESHOLD)
    print(f"Total merged segments: {len(merged_segments)}")

    if mode == "chunks":
        generate_chunks(merged_segments, mode)
        print("Video chunks generation completed.")
    elif mode == "full":
        # In 'full' mode, generate video chunks and a timeline for assembling the full video
        # Generate video chunks
        generate_chunks(merged_segments, mode)
        print("Video chunks generation completed.")

        # Additional steps for 'full' mode can be handled in the combining script
        # Since generating the full video requires precise alignment and handling of static images,
        # it's more efficient to manage it in the combining script.
        print("Note: In 'full' mode, assembling the complete video will be handled by the combining script.")

if __name__ == "__main__":
    main()
