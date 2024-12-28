from pyannote.audio import Pipeline
import speech_recognition as sr
from pydub import AudioSegment
import noisereduce as nr

# Function for noise reduction
def reduce_noise(input_audio_path, output_audio_path):
    audio = AudioSegment.from_wav(input_audio_path)
    audio = audio.set_channels(1)  # Convert to mono if stereo
    audio.export("mono_temp.wav", format="wav")  # Temporarily save as mono

    # Apply noise reduction
    y = nr.reduce_noise(y=audio.get_array_of_samples(), sr=audio.frame_rate)

    # Save the denoised audio
    denoised_audio = AudioSegment(
        y.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=1
    )
    denoised_audio.export(output_audio_path, format="wav")

# Load the pre-trained speaker diarization model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

# Path to your audio file
audio_file = "test1.wav"
denoised_audio_file = "denoised_test.wav"

# Apply noise reduction
reduce_noise(audio_file, denoised_audio_file)

# Diarization
diarization = pipeline(denoised_audio_file)

# Check if diarization is correctly processed
print("Diarization result:", diarization)

# Initialize the recognizer for speech-to-text
recognizer = sr.Recognizer()

# Define a function to transcribe speech
def transcribe_audio_segment(start_time, end_time, audio_file):
    with sr.AudioFile(audio_file) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source, duration=end_time-start_time, offset=start_time)
    
    try:
        # Use Google's speech recognition API to transcribe the speech
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        return "[Unintelligible]"
    except sr.RequestError:
        return "[API request failed]"

# Iterate over the diarization and transcribe each speaker's segment
for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker} speaks from {speech_turn.start:.2f} to {speech_turn.end:.2f}")
    transcription = transcribe_audio_segment(speech_turn.start, speech_turn.end, denoised_audio_file)
    print(f"Speaker {speaker}: {transcription}")
