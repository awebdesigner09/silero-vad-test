# /root/silero-vad-test/vad_server.py
import asyncio
import websockets
import torch
import os
import sys
import logging
import json # <-- Add this import

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Silero VAD Setup ---
VAD_MODEL = None
VAD_ITERATOR_CLASS = None
TARGET_SAMPLE_RATE = 16000  # Silero VAD expects 16kHz

def load_silero_vad_model():
    global VAD_MODEL, VAD_ITERATOR_CLASS
    try:
        # Try to find the local silero-vad-repo relative to this script
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        # Assuming silero-vad-repo is in the same directory as this script or one level up
        # Adjust these paths if your project structure is different
        possible_repo_paths = [
            os.path.join(current_script_path, 'silero-vad-repo'),
            os.path.join(current_script_path, '..', 'silero-vad-repo'),
            '/root/silero-vad-test/silero-vad-repo' # Default from original script
        ]
        
        silero_vad_repo_path = os.getenv('SILERO_VAD_REPO_PATH')
        if not silero_vad_repo_path:
            for path_attempt in possible_repo_paths:
                if os.path.isdir(path_attempt):
                    silero_vad_repo_path = path_attempt
                    logger.info(f"Found Silero VAD repo at: {silero_vad_repo_path}")
                    break
        
        if not silero_vad_repo_path or not os.path.isdir(silero_vad_repo_path):
            logger.error(f"Silero VAD repo path not found. Tried: {possible_repo_paths}. "
                         "Set SILERO_VAD_REPO_PATH or adjust script.")
            sys.exit(1)

        sys.path.insert(0, os.path.join(silero_vad_repo_path, 'src'))
        from silero_vad import utils_vad
        
        model_file_path = os.path.join(silero_vad_repo_path, 'src', 'silero_vad', 'data', 'silero_vad.jit')
        if not os.path.exists(model_file_path):
            logger.error(f"Silero VAD model file not found at: {model_file_path}")
            sys.exit(1)

        # Load the JIT model
        # Assuming utils_vad.init_jit_model returns only the model based on previous context
        VAD_MODEL = utils_vad.init_jit_model(model_file_path, device=torch.device('cpu')) # Specify device
        VAD_ITERATOR_CLASS = utils_vad.VADIterator 
        logger.info("Silero VAD model loaded successfully.")

    except ImportError:
        logger.exception("Failed to import Silero VAD. Ensure 'silero-vad-repo/src' is in sys.path or Silero VAD is installed.")
        sys.exit(1)
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
    
    # Each client gets its own VADIterator instance
    vad_iterator = VAD_ITERATOR_CLASS(VAD_MODEL, threshold=0.3, sampling_rate=TARGET_SAMPLE_RATE)
    
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
