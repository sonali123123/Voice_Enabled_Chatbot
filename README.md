Here is the README documentation for your Voice-Enabled Chatbot project:

---

# Voice-Enabled Chatbot

A real-time, voice-enabled AI chatbot that allows users to interact with a conversational AI system using speech. This project consists of two main components: the server-side application, built using FastAPI, and the client-side application, which handles audio recording, normalization, and playback.

## Features

- **Speech-to-Text (ASR):** Converts spoken input into text using Whisper.
- **Text-to-Speech (TTS):** Converts AI-generated text responses into audio using Google Text-to-Speech.
- **Context-Aware Responses:** Maintains a conversation history to provide contextually relevant answers.
- **Real-Time Interaction:** Designed for low-latency, real-time interactions.
- **REST API:** Server exposes an endpoint to process audio files and return audio responses.

---

## Technologies Used

- **Python:** Programming language for both server and client.
- **FastAPI:** Web framework for building the server API.
- **LangChain:** Framework for conversational AI and context-aware response generation.
- **Whisper:** Model for automatic speech recognition (ASR).
- **Google Text-to-Speech (gTTS):** For generating audio responses.
- **Pydub:** For audio processing on the client side.
- **Requests:** For HTTP communication between client and server.

---

## Architecture

1. **Client-Side (Client/client.py):**
   - Records audio using `speech_recognition` library.
   - Normalizes and prepares the audio for transmission.
   - Sends audio to the server and plays back the received response.

2. **Server-Side (Server/app.py):**
   - Processes incoming audio files.
   - Performs ASR using Whisper to transcribe audio into text.
   - Generates context-aware responses with a LangChain pipeline.
   - Converts text responses back into audio using TTS.
   - Sends the audio response back to the client.

---


## Usage

1. **Start the Server:**
   - Run the server application to expose the `/chat` endpoint.
   
2. **Start the Client:**
   - Run the client application to record audio, send it to the server, and play back the response.

3. **Interactive Voice Chat:**
   - Speak into the microphone when prompted by the client.
   - Listen to the AI-generated response.

---

## Endpoints

### POST `/chat`

- **Description:** Processes an audio file and returns an AI-generated audio response.
- **Request:**
  - File: Audio file (`audio/mp3` or `audio/wav`).
- **Response:**
  - Audio file (`audio/mpeg`).

---

## Key Features and Implementation

### 1. **Speech-to-Text (ASR):**
   - Utilizes OpenAI's Whisper model for transcription.
   - Optimized for English language processing.

### 2. **Context-Aware Responses:**
   - Implements LangChain’s retriever and chain mechanisms to maintain conversational history and provide contextually relevant answers.
   

### 3. **Text-to-Speech (TTS):**
   - Converts text responses into audio using Google TTS (`gTTS`).

### 4. **Client-Side Audio Processing:**
   - Records high-quality audio and normalizes volume using `Pydub`.
   - Handles playback of received audio responses.

---

## Future Improvements

- Add support for multiple languages.
- Implement real-time streaming for lower latency.
- Enhance error handling and retries on the client side.
- Incorporate a more robust TTS model for natural-sounding responses.

