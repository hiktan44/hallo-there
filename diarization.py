import argparse
from pyannote.audio import Pipeline

def run_diarization(access_token):
    # instantiate the pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=access_token
    )

    # run the pipeline on an audio file
    diarization = pipeline("audio/audio.wav")

    # dump the diarization output to disk using RTTM format
    with open("diarization/diarization.rttm", "w") as rttm:
        diarization.write_rttm(rttm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker diarization.")
    parser.add_argument("-access_token", required=True, help="Hugging Face access token")

    args = parser.parse_args()

    run_diarization(args.access_token)
