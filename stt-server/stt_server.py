import asyncio
import websockets
import whisper
import numpy as np
import torch
import logging
import os
import json

# Configure logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# STT Server Configuration
HOST = os.environ.get('HOST', 'localhost')
PORT = int(os.environ.get('PORT', 8766)) # Different port from VAD server
MODEL_NAME = os.environ.get('MODEL_NAME', "tiny.en") # e.g., "tiny.en", "base", "small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SAMPLE_RATE = 16000 # Should match client's audio sample rate

# Load Whisper model (once at startup)
model = None
logger.info(f"Attempting to load Whisper model: {MODEL_NAME} on {DEVICE}...")
try:
    model = whisper.load_model(MODEL_NAME, device=DEVICE)
    logger.info(f"Whisper model '{MODEL_NAME}' loaded successfully on {DEVICE}.")
except Exception as e:
    logger.error(f"Failed to load Whisper model '{MODEL_NAME}': {e}", exc_info=True)
    logger.error("STT server will not be able to process audio.")

async def stt_handler(websocket, path=None):
    client_address = websocket.remote_address
    logger.info(f"STT client connected: {client_address}")

    if not model:
        logger.error(f"Whisper model not loaded. Closing connection for {client_address}.")
        await websocket.send(json.dumps({"transcription": "", "error": "Whisper model not loaded on server."}))
        await websocket.close()
        return

    audio_bytes_buffer = bytearray()
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio_bytes_buffer.extend(message)
                # logger.debug(f"Received {len(message)} audio bytes from {client_address}. Buffer size: {len(audio_bytes_buffer)}")
            elif isinstance(message, str) and message == "END_OF_STREAM":
                logger.info(f"Received END_OF_STREAM from {client_address}. Processing {len(audio_bytes_buffer)} bytes.")
                if not audio_bytes_buffer:
                    logger.warning(f"Empty audio stream received from {client_address} before END_OF_STREAM.")
                    await websocket.send(json.dumps({"transcription": "", "error": "Empty audio received"}))
                    continue # Ready for next stream from this client

                # Convert Int16 bytes to Float32 NumPy array
                try:
                    audio_int16 = np.frombuffer(audio_bytes_buffer, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0 # Normalize to -1.0 to 1.0
                except Exception as e:
                    logger.error(f"Error converting audio data for {client_address}: {e}")
                    await websocket.send(json.dumps({"transcription": "", "error": f"Audio conversion error: {e}"}))
                    audio_bytes_buffer = bytearray() # Reset buffer
                    continue

                logger.info(f"Transcribing audio segment of {len(audio_float32) / TARGET_SAMPLE_RATE:.2f}s for {client_address}...")
                
                transcription_text = ""
                transcription_error = None
                try:
                    # Ensure audio is 1D
                    if audio_float32.ndim > 1:
                         audio_float32 = audio_float32.flatten()

                    # Set fp16=False if using CPU or if issues arise with fp16 on GPU
                    use_fp16 = DEVICE == "cuda" 
                    result = model.transcribe(audio_float32, fp16=use_fp16)
                    transcription_text = result["text"]
                    logger.info(f"Transcription for {client_address}: {transcription_text}")
                except Exception as e:
                    logger.error(f"Error during transcription for {client_address}: {e}", exc_info=True)
                    transcription_error = f"Transcription error: {e}"
                
                await websocket.send(json.dumps({"transcription": transcription_text, "error": transcription_error}))
                audio_bytes_buffer = bytearray() # Reset buffer for the next audio stream

            else:
                logger.warning(f"Received unexpected message type from {client_address}: {type(message)}")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"STT client {client_address} disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"STT client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error in STT handler for {client_address}: {e}", exc_info=True)
        try:
            # Attempt to inform client of server-side error
            if websocket.open:
                await websocket.send(json.dumps({"transcription": "", "error": f"Server error: {e}"}))
        except Exception as send_e:
            logger.error(f"Failed to send error message to client {client_address}: {send_e}")
    finally:
        logger.info(f"Cleaning up for STT client {client_address}.")
        # Ensure buffer is cleared if connection drops unexpectedly mid-stream
        audio_bytes_buffer = bytearray()


async def start_server():
    if not model:
        logger.error("Whisper model is not loaded. STT server cannot start.")
        return

    # Increased max_size for potentially larger audio segments. Adjust as needed.
    # 10MB limit; typical 16-bit PCM @ 16kHz is 32KB/sec. So 10MB is ~5 mins of audio.
    server = await websockets.serve(stt_handler, HOST, PORT, max_size=10 * 1024 * 1024) 
    logger.info(f"STT WebSocket server started on ws://{HOST}:{PORT}")
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("STT server shutting down...")
    except Exception as e:
        logger.critical(f"STT server failed to start or run: {e}", exc_info=True)
