import assemblyai as aai  # Ensure you're importing the assemblyai package

aai.settings.api_key = "52fb77d196dd4e269f9ad23fb1a60e7b"

# You can use a local filepath:
# audio_file = "./example.mp3"

# Or use a publicly-accessible URL:
audio_file = "test1.wav"

config = aai.TranscriptionConfig(
  speaker_labels=True,
)

transcript = aai.Transcriber().transcribe(audio_file, config)

for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")
