import os
import subprocess
import csv
import json

# Paths for folders
INPUT_FOLDER = "A"
OUTPUT_FOLDER = "B"
WHISPER_EXECUTABLE = "./build/bin/main"  # Adjust this path based on Whisper.cpp location

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def transcribe_file(file_path, output_csv_path):
    """
    Transcribes an audio or video file using Whisper.cpp and writes the transcription to a CSV.
    """
    try:
        # Run Whisper.cpp for transcription and output JSON
        json_output_path = output_csv_path.replace(".csv", ".json")
        command = [
            WHISPER_EXECUTABLE,
            "-f", file_path,
            "-o", json_output_path,
            "--output-json",  # Output transcription in JSON format
            "--timestamps", "true"
        ]
        subprocess.run(command, check=True)
        
        # Process JSON output to create the CSV
        with open(json_output_path, "r", encoding="utf-8") as json_file:
            transcription_data = json.load(json_file)

        # Write to CSV
        with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["speaker", "timestamp", "end time", "Wwrds"])  # Header row
            
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
