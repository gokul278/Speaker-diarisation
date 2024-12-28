import os
import wave
import json
from vosk import Model, KaldiRecognizer

# Path to Vosk model
model_path = "./vosk-model-small-en-us-0.15"  # Update with the model path

# Path to audio file
audio_file = "test.wav"  # Update with your audio file path

# Load the Vosk model
model = Model(model_path)

# Open the audio file
wf = wave.open(audio_file, "rb")

# Create a Kaldi recognizer with the model
recognizer = KaldiRecognizer(model, wf.getframerate())

# This will hold the results including words with speaker tag info (for diarization)
result = []

# Start processing the audio file
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if recognizer.AcceptWaveform(data):
        result.append(json.loads(recognizer.Result()))

# Process the final part
final_result = json.loads(recognizer.FinalResult())
result.append(final_result)

# Output the result (text with speaker diarization-like tags)
for res in result:
    if 'result' in res:
        for word in res['result']:
            print(f"word: '{word['word']}', speaker_tag: {word.get('speaker_tag', 'unknown')}")

# You can also save the result to a file if needed
with open("transcription_output.json", "w") as outfile:
    json.dump(result, outfile, indent=4)
    
    
    print(result)

# Optionally print the final output for inspection
print("\nFull Transcription with Speaker Tags:")
for res in result:
    if 'result' in res:
        for word in res['result']:
            print(f"Word: '{word['word']}', Speaker Tag: {word.get('speaker_tag', 'unknown')}")
