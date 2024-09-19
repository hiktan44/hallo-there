import os
import subprocess
import tempfile
from pydub import AudioSegment

# Configuration
AUDIO_FILE = "audio/input_audio.wav"          # Path to your main audio file
RTTM_FILE = "diarization/diarization.rttm"    # Path to your RTTM file
SOURCE_IMAGES_DIR = "source_images/"          # Directory containing source images
INFERENCE_SCRIPT = "inference.py"             # Path to Hallo's inference script
OUTPUT_VIDEOS_DIR = "output_videos/"          # Directory to save output videos
MERGE_GAP_THRESHOLD = 1.2                      # Maximum gap (in seconds) to merge segments
SPLIT_GAP = True                               # Whether to split gaps between segments

# Ensure output directory exists
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

# Initialize speaker pose counters
speaker_pose_counters = {}

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

    # Eliminate gaps by splitting them between segments
    if SPLIT_GAP:
        merged_segments = eliminate_gaps(merged_segments)

    return merged_segments

def eliminate_gaps(segments):
    """
    Adjust the start and end times of segments to eliminate gaps by splitting
    the gap time equally between the preceding and succeeding segments.

    Args:
        segments (list): List of merged segments.

    Returns:
        List of adjusted segments with no gaps.
    """
    adjusted_segments = segments.copy()
    for i in range(len(adjusted_segments) - 1):
        current_seg = adjusted_segments[i]
        next_seg = adjusted_segments[i + 1]
        gap = next_seg['start'] - current_seg['end']
        if gap > 0:
            half_gap = gap / 2
            # Adjust the end of the current segment
            adjusted_segments[i]['end'] += half_gap
            # Adjust the start of the next segment
            adjusted_segments[i + 1]['start'] -= half_gap
            print(f"Eliminated gap of {gap:.3f}s between chunk {i} and {i+1} by splitting {half_gap:.3f}s to each.")

    return adjusted_segments

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
    if speaker not in pose_counters:
        pose_counters[speaker] = 0
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

def main():
    # Parse and merge RTTM file
    merged_segments = parse_and_merge_rttm(RTTM_FILE, MERGE_GAP_THRESHOLD)
    print(f"Total merged segments: {len(merged_segments)}")

    for idx, segment in enumerate(merged_segments):
        speaker = segment['speaker']
        start = segment['start']
        end = segment['end']
        duration = end - start

        print(f"Processing chunk {idx}: Speaker={speaker}, Start={start:.3f}, Duration={duration:.3f} seconds")

        # Extract audio chunk
        try:
            audio_chunk_path = extract_audio_chunk(AUDIO_FILE, start, end)
        except Exception as e:
            print(f"Failed to extract audio chunk {idx}: {e}")
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
        output_video = os.path.join(OUTPUT_VIDEOS_DIR, f"chunk_{idx}_speaker_{speaker_id}.mp4")

        # Run inference
        try:
            run_inference(source_image, audio_chunk_path, output_video)
            print(f"Generated video: {output_video}")
        except Exception as e:
            print(f"Failed to generate video for chunk {idx}: {e}")
        finally:
            # Clean up temporary audio file
            os.unlink(audio_chunk_path)

if __name__ == "__main__":
    main()
