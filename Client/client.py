import requests
import wave
import time
import speech_recognition as sr
from pydub import AudioSegment, effects
from pydub.playback import play

# Server API URL
SERVER_URL = "http://10.7.0.28:6000/chat"

# Paths to audio feedback files
START_LISTENING_SOUND = "start_listening_sound.mp3"
STOPPED_LISTENING_SOUND = "stopped_listening_sound.mp3"

# Function to play sound using pydub
def play_sound(sound_file):
    sound = AudioSegment.from_mp3(sound_file)
    play(sound)

# Function to normalize audio for consistent volume levels
def normalize_audio(input_audio_path, output_audio_path):
    audio = AudioSegment.from_file(input_audio_path)
    normalized_audio = effects.normalize(audio)  # Normalize volume
    normalized_audio.export(output_audio_path, format="wav")  # Export as WAV

# Function to record high-quality audio
def record_high_quality_audio():
    recognizer = sr.Recognizer()
    audio_chunks = []
    silence_threshold = 3 # Maximum seconds of silence allowed to stop recording
    chunk_counter = 0

    print("Adjusting for ambient noise...")
    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Calibrate for ambient noise
        play_sound(START_LISTENING_SOUND)
        print("Listening...")

        try:
            while True:
                try:
                    print(f"Recording chunk {chunk_counter + 1}...")
                    # Capture a chunk of audio (10 seconds max)
                    audio = recognizer.listen(source, timeout=silence_threshold, phrase_time_limit=10)
                    audio_chunks.append(audio.get_wav_data())
                    chunk_counter += 1
                except sr.WaitTimeoutError:
                    if chunk_counter > 0:
                        print("Silence detected. Stopping recording.")
                        break
                    else:
                        play_sound(STOPPED_LISTENING_SOUND)
                        print("No speech detected, please try again.")
                        return None

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

        play_sound(STOPPED_LISTENING_SOUND)
        print(f"Recording completed with {chunk_counter} chunks.")

        # Save raw audio chunks to a WAV file
        raw_audio_file = "raw_audio.wav"
        with wave.open(raw_audio_file, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit samples
            wf.setframerate(16000)  # 16 kHz sampling rate
            for chunk in audio_chunks:
                wf.writeframes(chunk)

        # Normalize and save as final audio
        normalized_audio_file = "input_audio.wav"
        normalize_audio(raw_audio_file, normalized_audio_file)
        print("Audio normalized and saved.")
        return normalized_audio_file

# Send the recorded audio to the server and receive response
def send_to_server(audio_file_path):
    max_retries = 3
    for attempt in range(max_retries):
        print(f"Sending audio to server... (Attempt {attempt + 1}/{max_retries})")
        try:
            with open(audio_file_path, 'rb') as f:
                response = requests.post(SERVER_URL, files={'audio_file': f})
            if response.status_code == 200:
                response_audio_path = "response_audio.mp3"
                with open(response_audio_path, 'wb') as audio_file:
                    audio_file.write(response.content)
                print("Response received.")
                return response_audio_path
            else:
                print(f"Server error: {response.status_code} - {response.text}")
                return None
        except requests.ConnectionError:
            print(f"Connection error. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(1)
    print("Failed to connect to the server after multiple attempts.")
    return None

# Play the response audio directly
def play_audio(response_audio_path):
    if response_audio_path:
        print("Playing server response...")
        audio = AudioSegment.from_file(response_audio_path)
        play(audio)
    else:
        print("No audio to play.")

if __name__ == "__main__":
    while True:
        # Step 1: Record high-quality audio
        audio_file_path = record_high_quality_audio()
        if not audio_file_path:
            continue  # Restart loop if no audio was captured

        # Step 2: Send recorded audio to the server and get response
        print("Processing your request, please wait...")
        response_audio_path = send_to_server(audio_file_path)

        # Step 3: Play the received audio response
        play_audio(response_audio_path)