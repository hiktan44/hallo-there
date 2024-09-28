import os
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    concatenate_videoclips,
    CompositeVideoClip,
    clips_array,
    vfx,
    AudioFileClip,
    afx
)

# Paths
diarization_file = 'diaraization/diaraization_index.rttm'
video_chunks_path = 'output_videos'
static_images_path = 'source_images'
audio_file = "audio/input_audio.wav"

# Constants
NUM_POSES = 4  # Number of poses per speaker
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 512
CLIP_WIDTH = OUTPUT_WIDTH // 2  # Each clip is half the output width
ADJUST_AUDIO = False

# Read the diarization file and extract events
events = []
with open(diarization_file, 'r') as f:
    for line in f:
        tokens = line.strip().split()
        index = int(tokens[0])
        start_time = float(tokens[4])
        duration = float(tokens[5])
        end_time = start_time + duration
        speaker = tokens[8]
        events.append({
            'index': index,
            'start_time': start_time,
            'duration': duration,
            'end_time': end_time,
            'speaker': speaker
        })

# Determine total duration
total_duration = max(event['end_time'] for event in events)

# Build speaker intervals
speakers = set(event['speaker'] for event in events)
speaker_intervals = {speaker: [] for speaker in speakers}
for event in events:
    speaker_intervals[event['speaker']].append(event)

# Process each speaker
speaker_final_clips = {}
for speaker in speakers:
    # Sort intervals
    intervals = sorted(speaker_intervals[speaker], key=lambda x: x['start_time'])
    clips = []
    last_end = 0.0
    pose_counter = 0
    for event in intervals:
        start = event['start_time']
        end = event['end_time']
        duration = event['duration']
        index = event['index']
        speaker_lower = speaker.lower()

        # Handle non-speaking intervals
        if last_end < start:
            silence_duration = start - last_end
            pose_index = pose_counter % NUM_POSES
            pose_counter += 1
            image_path = os.path.join(static_images_path, f'{speaker}_pose_{pose_index}.png')
            image_clip = ImageClip(image_path, duration=silence_duration)
            fade_duration = min(0.5, silence_duration)
            image_clip = image_clip.fx(vfx.fadein, fade_duration)
            # Ensure the static image has silent audio
            image_clip = image_clip.set_audio(None)
            clips.append(image_clip)

        # Handle speaking intervals (video chunks)
        video_filename = f'chunk_{index}_{speaker_lower}.mp4'
        video_path = os.path.join(video_chunks_path, video_filename)
        if os.path.exists(video_path):
            video_clip = VideoFileClip(video_path).subclip(0, duration)
            if not ADJUST_AUDIO:
                video_clip = video_clip.without_audio()
            clips.append(video_clip)
        else:
            print(f'Warning: Video file {video_path} not found.')
        last_end = end

    # Handle any remaining time after the last interval
    if last_end < total_duration:
        silence_duration = total_duration - last_end
        pose_index = pose_counter % NUM_POSES
        pose_counter += 1
        image_path = os.path.join(static_images_path, f'{speaker}_pose_{pose_index}.png')
        image_clip = ImageClip(image_path, duration=silence_duration)
        fade_duration = min(0.5, silence_duration)
        image_clip = image_clip.fx(vfx.fadein, fade_duration)
        image_clip = image_clip.set_audio(None)
        clips.append(image_clip)

    # Concatenate all clips for the speaker
    final_clip = concatenate_videoclips(clips, method='compose')
    final_clip = final_clip.resize((CLIP_WIDTH, OUTPUT_HEIGHT))
    speaker_final_clips[speaker] = final_clip

# Ensure both speaker clips have the same duration
durations = [clip.duration for clip in speaker_final_clips.values()]
max_duration = max(durations)
for speaker, clip in speaker_final_clips.items():
    if clip.duration < max_duration:
        # Extend the clip by freezing the last frame
        freeze_frame = clip.to_ImageClip(clip.duration - 1/clip.fps)
        freeze_clip = freeze_frame.set_duration(max_duration - clip.duration)
        freeze_clip = freeze_clip.set_audio(None)
        speaker_final_clips[speaker] = concatenate_videoclips([clip, freeze_clip])

# Arrange speakers side by side
speakers_sorted = sorted(speaker_final_clips.keys())
left_clip = speaker_final_clips[speakers_sorted[0]]
right_clip = speaker_final_clips[speakers_sorted[1]]
final_video = clips_array([[left_clip, right_clip]])

# Set the final video size
final_video = final_video.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT))

# Load the main audio
try:
    main_audio = AudioFileClip(audio_file)
except Exception as e:
    print(f"Error loading main audio file: {e}")

# Set the audio to the combined video
final_video = final_video.set_audio(main_audio)

# Write the output video file
final_video.write_videofile('final_video.mp4', codec='libx264', audio_codec='aac')