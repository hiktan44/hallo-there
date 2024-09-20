*Hallo There* is a combination of tools to generate realistic talking head video of a multi-person audio file,
also known as lipsyncing.

<video width="512" height="256" controls>
  <source src="https://github.com/abrakjamson/hallo-there/blob/mainexamples/hallo_there_short.mp4
  " type="video/mp4">
  Your browser does not support the video tag.
</video>

I created it because I found Google's NotebookLM podcast audio feature, which produces fantastic audio 
quality. I've been interested in AI-generated images for years and saw the opportunity to combine them.

The major tools are Hallo and speaker-diarization-3.1, as well as your preferred image generation tool.

This project is extremely barebones. You'll need to be at least a little familiar with Python environment 
management and dependency installation. You'll also want a video card with at least 8GB of video memory - 
16GB may be better. Even still, expect it to take about 2-3 minutes of processing on an A40 per second of video!

# Setup

## Install Summary
The basic idea for installation and use is:
1. Get your audio file
2. Use speaker-diarization-3.1
3. Generate or get source images
4. Generate the video clips with Hallo
5. Combine the video clips

## Installation Step Detail
1. Create and activate a Conda environment
    1. If you need it, go install miniconda
2. Clone this repo, or download the .zip and unzip it
3. Add source images and audio, see the Prepare Inference Data section
4. Create diarization file
    1. Navigate to the root directory of the project
    2. Install pyannote.audio 3.1 with pip install pyannote.audio
    3. Accept pyannote/segmentation-3.0 user conditions at Huggingface
    4. Accept pyannote/speaker-diarization-3.1 user conditions at Huggingface
    5. Create access token at hf.co/settings/tokens.
    6. python diarization.py -access_token the_token_you_generated_on_huggingface
5. Install hallo and prebuilt packages
    1. pip install -r requirements.txt
    2. pip install .
    3. Install ffmpeg
        1. (Linux) apt-get install ffmpeg
        2. (Windows) Install from https://ffmpeg.org/download.html and add it to your system path variable
    4. Get the pretrained models
        1. git lfs install
        2. git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models
        3. alternately, view the hallo repo's readme for each of the models you need

I've tested on Windows 11 with a GeForce 3060 12GB and Ubuntu 22.04 Linux with an A40.
If these steps are completely unintelligible, that's OK! You'll have a better time using one of 
the paid and proprietary services to do this. Check out HeyGen, Hydra, or LiveImageAI. Or have 
an AI walk you through the steps if you'd like to learn a new skill!

# Run

##  Prepare Inference Data

A sample is included if you want to try it out now. Otherwise, *Hallo There* has a few simple 
requirements for input data:

For the source images:

1. It should be cropped into squares. If it isn't 512x512, it will be resized.
2. The face should be the main focus, making up 50%-70% of the image.
3. The face should be facing forward, with a rotation angle of less than 30° (no side profiles).
4. There should be four poses for each of the two speakers.

For the driving audio:

1. It must be in WAV format.
2. It must be in English since the training datasets are only in this language.
3. Ensure the vocals are clear; background music is acceptable.

You'll need to add your files to these directories:
```
project/
│
├── source_images/
│   ├── SPEAKER_00_pose_0.png
│   ├── SPEAKER_00_pose_1.png
│   ├── SPEAKER_00_pose_2.png
│   ├── SPEAKER_00_pose_3.png
│   ├── SPEAKER_01_pose_0.png
│   ├── SPEAKER_01_pose_1.png
│   ├── SPEAKER_01_pose_2.png
│   └── SPEAKER_01_pose_3.png
│
├── audio/
│   └── input_audio.wav
│
├── diarization/
│   └── diarization.rttm
│
└── output_videos/
```

## Generate
Once you've complted the install and prepared the speaker pose images, audio, and diarization files, 
run generate_videos.py
```
generate_videos.py
```
You can specify -mode full to generate slight head movements while an avatar is not speaking, at the cost of
double the runtime.

After a lot of time and a lot of console output, you'll get the chunks of video in the output_videos folder. 
Next run combine_videos.py.
```
combine_videos.py
```
This will also take some time, but not nearly as much as creating the chunks. The project root will contain
final_combined_output.mp4

## Configuration options
To be documented. You can see them in the main python scripts.

# Remaining work
Work to be done:
- ✅ Proof-of-concept
- ☑️ Add examples
- ✅ Installation instructions
- ☑️ Document configuration options
- ☑️ Replace use of SD 1.5 with Flux schnell

*Hallo There* is licensed under MIT. Thank you to the Hallo team for doing the heavy lifting on this project!