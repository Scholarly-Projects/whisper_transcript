import os
import subprocess
import csv
import json
import ffmpeg
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio import Model
from tempfile import NamedTemporaryFile

# Paths for folders
INPUT_FOLDER = "A"
OUTPUT_FOLDER = "B"
WHISPER_EXECUTABLE = "./whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/Users/andrewweymouth/Documents/GitHub/whisper_transcript/whisper.cpp/models/ggml-large-v3.bin"
DIARIZATION_MODEL = "pyannote/speaker-diarization"
INITIAL_PROMPT = "This is an oral history interview. The speakers are [Insert Names]. They discuss historical events and personal experiences."

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def convert_mp3_to_wav(mp3_path, wav_path, sample_rate=16000):
    """ Converts an MP3 file to WAV format using ffmpeg-python. """
    try:
        ffmpeg.input(mp3_path).output(wav_path, ar=sample_rate).overwrite_output().run()
        print(f"Converted {mp3_path} to {wav_path} with sample rate {sample_rate} Hz")
    except ffmpeg.Error as e:
        print(f"Error converting {mp3_path} to WAV: {e}")
        raise

def apply_speaker_diarization(audio_path):
    """ Runs Pyannote speaker diarization on the given audio file. """
    model = Model.from_pretrained(DIARIZATION_MODEL)
    pipeline = SpeakerDiarization(segmentation=model)
    
    diarization = pipeline(audio_path)
    speaker_segments = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments[(turn.start, turn.end)] = speaker

    return speaker_segments

def transcribe_file(file_path, output_csv_path, sample_rate=16000):
    """ Transcribes an audio file using Whisper.cpp and adds speaker diarization. """
    try:
        # Convert MP3 to WAV if needed
        if file_path.lower().endswith(".mp3"):
            wav_file_path = file_path.replace(".mp3", ".wav")
            convert_mp3_to_wav(file_path, wav_file_path, sample_rate)
            file_path = wav_file_path  

        # Output JSON path
        json_output_path = output_csv_path.replace(".csv", ".json")

        # Run Whisper.cpp transcription
        command = [
            WHISPER_EXECUTABLE,
            "-f", file_path,
            "-of", json_output_path,
            "--output-json",
            "--model", MODEL_PATH,
            "--word_timestamps",
            "--initial_prompt", INITIAL_PROMPT
        ]
        
        subprocess.run(command, check=True)

        if os.path.exists(json_output_path + ".json"):
            os.rename(json_output_path + ".json", json_output_path)
        if not os.path.exists(json_output_path):
            raise FileNotFoundError(f"Expected JSON output file not found: {json_output_path}")

        # Load transcription results
        with open(json_output_path, "r", encoding="utf-8") as json_file:
            transcription_data = json.load(json_file)

        # Apply Pyannote diarization
        speaker_segments = apply_speaker_diarization(file_path)

        # Open CSV for writing
        with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Speaker", "Start Time", "End Time", "Words"])

            for segment in transcription_data.get("transcription", []):
                start_time = segment['timestamps']['from']
                end_time = segment['timestamps']['to']
                text = segment['text']

                # Assign speaker based on closest diarization match
                speaker = "Unknown"
                for (seg_start, seg_end), assigned_speaker in speaker_segments.items():
                    if seg_start <= start_time <= seg_end:
                        speaker = assigned_speaker
                        break

                writer.writerow([speaker, start_time, end_time, text])

        os.remove(json_output_path)
        print(f"Transcription completed: {output_csv_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_folder(input_folder, output_folder, sample_rate=16000):
    """ Processes all audio files in the input folder. """
    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(input_file_path) and file_name.lower().endswith((".mp3", ".mp4", ".wav", ".m4a")):
            output_file_name = os.path.splitext(file_name)[0] + ".csv"
            output_file_path = os.path.join(output_folder, output_file_name)
            transcribe_file(input_file_path, output_file_path, sample_rate)

if __name__ == "__main__":
    print("Starting transcription tool...")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    print("All files have been processed.")
