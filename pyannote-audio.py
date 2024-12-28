from pyannote.audio import Pipeline
import torch
import whisper
from pydub import AudioSegment

# Load the PyAnnote diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

# Load Whisper model for transcription (use 'base', 'small', 'medium', 'large' based on your needs)
whisper_model = whisper.load_model("large")

# Apply the diarization pipeline to your audio file
diarization = pipeline("test1.wav")

# Load the audio file using pydub
audio = AudioSegment.from_wav("test1.wav")

# Iterate over diarization results and transcribe each segment
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start = turn.start * 1000  # Convert to milliseconds
    end = turn.end * 1000

    # Extract the segment based on start and end time
    segment = audio[start:end]

    # Save the segment as a temporary file
    segment.export("temp_segment.wav", format="wav")

    # Perform transcription using Whisper
    result = whisper_model.transcribe("temp_segment.wav")

    # Print the speaker and the transcribed text
    print(f"Speaker {speaker}: {result['text'].strip()}")
