# /root/silero-vad-test/vad_server.py
import asyncio
import websockets
import torch
import os
import logging
import json # <-- Add this import
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv() 

# --- Logging Setup ---
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Silero VAD Setup ---
VAD_MODEL = None
VAD_ITERATOR_CLASS = None
TARGET_SAMPLE_RATE = 16000  # Silero VAD expects 16kHz

# --- VAD Configuration (from environment or defaults) ---
VAD_THRESHOLD = float(os.getenv('VAD_THRESHOLD', 0.6)) 
VAD_MIN_SILENCE_DURATION_MS = int(os.getenv('VAD_MIN_SILENCE_DURATION_MS', 300)) # Increased from 100ms
VAD_SPEECH_PAD_MS = int(os.getenv('VAD_SPEECH_PAD_MS', 200)) # Increased default from 30ms
VAD_MIN_SPEECH_DURATION_MS = int(os.getenv('VAD_MIN_SPEECH_DURATION_MS', 250))

def load_silero_vad_model():
    global VAD_MODEL, VAD_ITERATOR_CLASS
    try:
        logger.info("Attempting to load Silero VAD model using torch.hub.load...")
        # `force_reload=True` can be useful for debugging or ensuring the latest version,
        # but for production, `force_reload=False` (default) is usually better to use cached versions.
        # `trust_repo=True` is required for this specific repository.
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False, # Set to True to always re-download
                                      trust_repo=True)
        VAD_MODEL = model
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils
        VAD_ITERATOR_CLASS = VADIterator
        logger.info("Silero VAD model loaded successfully.")
    except Exception as e:
        logger.exception(f"Error during Silero VAD setup: {e}")
        sys.exit(1)

# --- WebSocket Handler ---
async def vad_audio_processor(websocket, path=None):
    if VAD_MODEL is None or VAD_ITERATOR_CLASS is None:
        logger.error("VAD model not loaded. Cannot process audio.")
        await websocket.close(code=1011, reason="VAD model not available")
        return

    client_address = websocket.remote_address
    logger.info(f"Client connected: {client_address}")
    
    logger.info(f"Initializing VADIterator for {client_address} with: "
                f"threshold={VAD_THRESHOLD}, "
                f"min_silence_ms={VAD_MIN_SILENCE_DURATION_MS}, speech_pad_ms={VAD_SPEECH_PAD_MS}, "
                f"min_speech_ms={VAD_MIN_SPEECH_DURATION_MS}")
    # Each client gets its own VADIterator instance
    vad_iterator = VAD_ITERATOR_CLASS(VAD_MODEL, 
                                      threshold=VAD_THRESHOLD, 
                                      sampling_rate=TARGET_SAMPLE_RATE,
                                      min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS, # VADIterator uses this
                                      speech_pad_ms=VAD_SPEECH_PAD_MS) # VADIterator uses this
    
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Assuming client sends raw Int16 PCM audio data
                # Each sample is 2 bytes.
                
                # Convert bytes to Int16 tensor
                audio_int16 = torch.frombuffer(message, dtype=torch.int16)
                
                # Normalize to Float32 tensor, range [-1, 1]
                audio_float32 = audio_int16.float() / 32767.0 # Max Int16 value is 32767

                # Ensure tensor is 1D for VADIterator
                if audio_float32.ndim > 1:
                    audio_float32 = audio_float32.squeeze()
                
                if audio_float32.numel() == 0: # Check if tensor is empty
                    logger.warning(f"[{client_address}] Received empty audio buffer, skipping.")
                    continue
                
                # Get VAD prediction
                speech_dict = vad_iterator(audio_float32, return_seconds=False)

                response_message = {"type": "VAD_STATUS", "message": "no_speech"} # Default

                if speech_dict:
                    if 'start' in speech_dict:
                        response_message = {"type": "VAD_SPEECH_START"}
                        logger.info(f"[{client_address}] Speech Start (frame: {speech_dict['start']})")
                    elif 'end' in speech_dict:
                        response_message = {"type": "VAD_SPEECH_END"}
                        logger.info(f"[{client_address}] Speech End (frame: {speech_dict['end']})")
                        vad_iterator.reset_states() # Reset for next utterance detection
                    else:
                        # This case implies speech is detected in the current chunk but it's not a new 'start'
                        # or 'end' event from the iterator's perspective for this specific chunk.
                        # You might want to send a general "speech_ongoing" status or stick to "no_speech"
                        # if the VADIterator only gives explicit start/end.
                        # For simplicity, let's assume if speech_dict is not empty and not start/end,
                        # it's ongoing speech. The client might not explicitly use "VAD_SPEECH_ONGOING".
                        response_message = {"type": "VAD_STATUS", "message": "speech_ongoing"}
                
                await websocket.send(json.dumps(response_message))
            else:
                logger.warning(f"[{client_address}] Received non-binary message: {message}")
                await websocket.send(json.dumps({"type": "ERROR", "message": "Expected binary audio data."}))

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_address} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.info(f"Client {client_address} disconnected with error: {e}")
    except Exception as e:
        logger.exception(f"Error in VAD handler for {client_address}: {e}")
    finally:
        logger.info(f"Cleaning up for client {client_address}.")
        vad_iterator.reset_states() # Ensure reset on any disconnect

async def start_server(host="0.0.0.0", port=8765):
    load_silero_vad_model() # Load model once at startup
    if VAD_MODEL is None:
        logger.error("Server cannot start: VAD model failed to load.")
        return

    logger.info(f"Starting Silero VAD WebSocket server on ws://{host}:{port}")
    async with websockets.serve(vad_audio_processor, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("\nServer shutting down...")
    except Exception as e:
        logger.exception("Failed to start server.")
