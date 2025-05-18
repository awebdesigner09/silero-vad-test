# Real-time Voice Activity Detection (VAD) System

This project implements a real-time Voice Activity Detection system using a React JS frontend and a Python backend powered by Silero VAD.

## Architecture

The system consists of two main components:

1.  **React JS Frontend (Client):**
    *   Runs in the user's browser.
    *   Captures audio from the microphone using `navigator.mediaDevices.getUserMedia`.
    *   Streams the captured audio chunks (16-bit PCM at 16kHz) to the Python backend via WebSockets.
    *   Receives VAD status updates (`speech_start`, `speech_end`, `speech_ongoing`, `no_speech`) from the backend.
    *   Displays the current VAD status to the user.

2.  **Python WebSocket Backend (Server):**
    *   Runs a WebSocket server (e.g., using the `websockets` library).
    *   Receives audio chunks from the React client.
    *   Uses the pre-trained Silero VAD model to process each audio chunk.
    *   Determines if the chunk contains speech, the start of speech, or the end of speech.
    *   Sends the VAD status back to the connected React client.

## How VAD Works

### 1. Microphone Audio Detection & Transmission

*   **Client-Side Capture:** The React application, running in the browser, requests access to the user's microphone.
*   **Audio Streaming:** Once permission is granted, the client captures audio in small chunks. These chunks are typically processed to match the format expected by the VAD model (e.g., 16kHz, 16-bit PCM, mono).
*   **WebSocket Transmission:** Each audio chunk is sent as binary data over a WebSocket connection to the Python backend server.

### 2. Server-Side Voice Activity Detection

The Python server (`vad_server.py`) performs the core VAD processing:

*   **Receiving Audio:** The server listens for incoming WebSocket messages containing audio data.
*   **Data Preparation:**
    *   The received raw audio bytes are converted into a PyTorch tensor (e.g., `torch.int16`).
    *   This tensor is normalized to a 32-bit floating-point tensor with values between -1.0 and 1.0, as required by the Silero VAD model.
*   **Silero VAD Processing:**
    *   An instance of `VADIterator` (from Silero VAD utilities) is used for each connected client to process the stream of audio chunks.
    *   The `VADIterator` takes the prepared audio tensor and returns a dictionary indicating speech events:
        *   `{'start': frame_index}`: When speech is detected to start.
        *   `{'end': frame_index}`: When detected speech ends.
        *   `None` or other dictionary: If speech is ongoing within the chunk or no speech is detected.
*   **Interpreting Results:** Based on the output from `VADIterator`, the server determines a status:
    *   `"speech_start"`: If the start of speech is detected.
    *   `"speech_end"`: If the end of speech is detected. The VAD iterator's state is typically reset after this.
    *   `"speech_ongoing"`: If speech is detected within the current chunk but it's not a new start or end.
    *   `"no_speech"`: If no speech is detected.
*   **Sending Status to Client:** This determined `vad_status` is sent back to the React client via the WebSocket connection.

### 3. Noise Handling

*   **Model Capability:** The Silero VAD model is a neural network trained to distinguish speech from various types of non-speech sounds (noise). Its effectiveness depends on the training data and model architecture.
*   **Thresholding:** The `VADIterator` is initialized with a `threshold` parameter (e.g., 0.5). This threshold is applied to the model's internal speech probability score.
    *   A higher threshold makes the VAD less sensitive (more noise robust, but might miss quiet speech).
    *   A lower threshold makes the VAD more sensitive (catches quieter speech, but might misclassify some noise as speech).
    This threshold can be tuned based on the operating environment.

## Integration with Speech-to-Text (STT)

The VAD status messages are crucial for efficient STT integration:

1.  **Buffering:** When the client receives `"speech_start"`, it (or the server) can begin buffering the subsequent audio chunks.
2.  **Segment Collection:** Audio collection continues as long as the status is `"speech_ongoing"` or until `"speech_end"` is received.
3.  **Sending to STT:** The complete audio segment (from start to end of speech) is then sent to an STT engine for transcription.

This approach ensures that only relevant speech segments are processed by the STT engine, improving efficiency, reducing costs (for cloud STT services), and potentially enhancing STT accuracy.

## Running the System

1.  **Start the Python VAD Server:**
    ```bash
    python /path/to/your/vad_server.py
    ```
2.  **Start the React Client:**
    Navigate to the client project directory (`/root/silero-vad-test/vad-client/`) and run:
    ```bash
    yarn dev
    # or
    # npm start
    ```
    Open the application in your browser (usually `http://localhost:3000`).