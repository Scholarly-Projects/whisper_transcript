import os
import subprocess
import csv
import json
from pydub import AudioSegment

# Paths for folders
INPUT_FOLDER = "A"
OUTPUT_FOLDER = "B"
WHISPER_EXECUTABLE = "./whisper.cpp/build/bin/whisper-cli"  # Corrected path

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Converts an MP3 file to WAV format using pydub and resamples to 16 kHz.
    """
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        # Resample to 16 kHz (required by Whisper)
        audio = audio.set_frame_rate(16000)
        audio.export(wav_path, format="wav")
        print(f"Converted {mp3_path} to {wav_path} and resampled to 16 kHz")
    except Exception as e:
        print(f"Error converting {mp3_path} to WAV: {e}")
        raise

def transcribe_file(file_path, output_csv_path):
    """
    Transcribes an audio or video file using Whisper.cpp and writes the transcription to a CSV.
    """
    try:
        # If the file is an MP3, convert it to WAV first
        if file_path.lower().endswith(".mp3"):
            wav_file_path = file_path.replace(".mp3", ".wav")
            convert_mp3_to_wav(file_path, wav_file_path)
            file_path = wav_file_path  # Use the WAV file for transcription

        # Output JSON path (update to match whisper-cli options)
        json_output_path = output_csv_path.replace(".csv", ".json")
        
        # Command to run whisper-cli with correct arguments
        command = [
            WHISPER_EXECUTABLE,
            "-f", file_path,  # Input file (WAV or other supported formats)
            "-of", json_output_path,  # Corrected argument for output file path
            "--output-json",  # Output transcription in JSON format
            "--no-timestamps",  # Remove timestamp flag, as it's not supported
            "--model", "/Users/andrewweymouth/Documents/GitHub/whisper_transcript/whisper.cpp/models/ggml-base.en.bin"  # Full path to the model
        ]
        
        subprocess.run(command, check=True)

        # Ensure the JSON file is created
        if not os.path.exists(json_output_path):
            raise FileNotFoundError(f"Expected JSON output file not found: {json_output_path}")
        
        # Process JSON output to create the CSV
        with open(json_output_path, "r", encoding="utf-8") as json_file:
            transcription_data = json.load(json_file)

        # Write transcription to CSV
        with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["speaker", "timestamp", "end time", "Words"])  # Header row
            
            for segment in transcription_data["segments"]:
                speaker = segment.get("speaker", "Speaker 1")  # Assign "Speaker 1" by default
                timestamp = segment["start"]
                end_time = segment["end"]
                words = segment["text"]
                writer.writerow([speaker, timestamp, end_time, words])

        # Clean up temporary JSON file
        os.remove(json_output_path)
        print(f"Transcription completed: {output_csv_path}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_folder(input_folder, output_folder):
    """
    Processes all audio/video files in the input folder and generates transcriptions in the output folder.
    """
    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(input_file_path) and file_name.lower().endswith((".mp3", ".mp4", ".wav", ".m4a")):
            output_file_name = os.path.splitext(file_name)[0] + ".csv"
            output_file_path = os.path.join(output_folder, output_file_name)
            transcribe_file(input_file_path, output_file_path)

if __name__ == "__main__":
    print("Starting transcription tool...")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    print("All files have been processed.")