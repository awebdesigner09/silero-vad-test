import asyncio
import websockets
import whisper
import numpy as np
import torch
import logging
import os
import json
import functools # Import functools
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
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
TURN_SILENCE_TIMEOUT_SECONDS = float(os.environ.get('TURN_SILENCE_TIMEOUT_SECONDS', 2.0)) # Seconds of silence to trigger end of turn

# Load Whisper model (once at startup)
model = None
logger.info(f"Attempting to load Whisper model: {MODEL_NAME} on {DEVICE}...")
try:
    model = whisper.load_model(MODEL_NAME, device=DEVICE)
    logger.info(f"Whisper model '{MODEL_NAME}' loaded successfully on {DEVICE}.")
except Exception as e:
    logger.error(f"Failed to load Whisper model '{MODEL_NAME}': {e}", exc_info=True)
    logger.error("STT server will not be able to process audio.")

async def _transcribe_and_send_audio(audio_data_bytes, websocket_conn, client_addr):
    """
    Converts audio bytes, transcribes using Whisper, and sends the result.
    """
    if not model:
        logger.error(f"Whisper model not loaded. Cannot transcribe for {client_addr}.")
        try:
            await websocket_conn.send(json.dumps({"transcription": "", "error": "Whisper model not loaded on server."}))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed for {client_addr} before sending 'model not loaded' error.")
        return

    if not audio_data_bytes:
        logger.warning(f"_transcribe_and_send_audio called with empty buffer for {client_addr}.")
        # It's up to the caller to decide if an "empty audio" error message is sent to client.
        # This function just returns if there's no data to process.
        return

    logger.info(f"Preparing to transcribe {len(audio_data_bytes)} bytes for {client_addr}.")
    try:
        audio_int16 = np.frombuffer(audio_data_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0 # Normalize to -1.0 to 1.0
    except Exception as e:
        logger.error(f"Error converting audio data for {client_addr}: {e}")
        try:
            await websocket_conn.send(json.dumps({"transcription": "", "error": f"Audio conversion error: {e}"}))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed for {client_addr} before sending 'audio conversion error'.")
        return

    logger.info(f"Transcribing audio segment of {len(audio_float32) / TARGET_SAMPLE_RATE:.2f}s for {client_addr}...")
    transcription_text = ""
    transcription_error = None
    try:
        # Ensure audio is 1D
        if audio_float32.ndim > 1:
            audio_float32 = audio_float32.flatten()

        if audio_float32.size == 0:
            logger.warning(f"Audio data became empty after processing for {client_addr}. Original bytes: {len(audio_data_bytes)}")
            try:
                await websocket_conn.send(json.dumps({"transcription": "", "error": "Processed audio data is empty"}))
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Connection closed for {client_addr} before sending 'processed audio empty' error.")
            return

        # Set fp16=False if using CPU or if issues arise with fp16 on GPU
        use_fp16 = DEVICE == "cuda"
        # For long transcriptions, consider running in an executor to avoid blocking asyncio loop:
        loop = asyncio.get_running_loop()
        # Use functools.partial to correctly pass keyword arguments to model.transcribe
        # model.transcribe expects audio as the first positional argument, and fp16 as a keyword argument.
        transcribe_func = functools.partial(model.transcribe, audio=audio_float32, fp16=use_fp16)
        result = await loop.run_in_executor(None, transcribe_func)
        transcription_text = result["text"]
        logger.info(f"Transcription for {client_addr}: {transcription_text}")
    except Exception as e:
        logger.error(f"Error during transcription for {client_addr}: {e}", exc_info=True)
        transcription_error = f"Transcription error: {e}"
    
    try:
        await websocket_conn.send(json.dumps({"transcription": transcription_text, "error": transcription_error}))
    except websockets.exceptions.ConnectionClosed:
        logger.warning(f"WebSocket connection for {client_addr} is closed. Cannot send transcription results.")
        if transcription_text or transcription_error:
            logger.info(f"Unsent transcription for {client_addr}: Text='{transcription_text}', Error='{transcription_error}'")
async def stt_handler(websocket, path=None):
    client_address = websocket.remote_address
    logger.info(f"STT client connected: {client_address}")

    if not model:
        logger.error(f"Whisper model not loaded. Closing connection for {client_address}.")
        try:
            await websocket.send(json.dumps({"transcription": "", "error": "Whisper model not loaded on server."}))
        except websockets.exceptions.ConnectionClosed: # Catch if connection is already closed
            logger.warning(f"Connection already closed for {client_address} when trying to send 'model not loaded' error.")
            pass # Client might have already closed
        finally:
            await websocket.close()
        return

    audio_bytes_buffer = bytearray()
    try:
        while True: # Loop to handle multiple turns or messages
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=TURN_SILENCE_TIMEOUT_SECONDS)

                if isinstance(message, bytes):
                    audio_bytes_buffer.extend(message)
                    # logger.debug(f"Received {len(message)} audio bytes. Buffer: {len(audio_bytes_buffer)}")
                elif isinstance(message, str) and message == "END_OF_STREAM":
                    logger.info(f"END_OF_STREAM from {client_address}. Processing buffer ({len(audio_bytes_buffer)} bytes).")
                    if audio_bytes_buffer:
                        await _transcribe_and_send_audio(audio_bytes_buffer, websocket, client_address)
                        audio_bytes_buffer = bytearray()
                    else:
                        logger.warning(f"Empty audio on END_OF_STREAM from {client_address}.")
                        try:
                            await websocket.send(json.dumps({"transcription": "", "error": "Empty audio received with END_OF_STREAM"}))
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"Connection closed for {client_address} before sending 'empty audio on EOS' message.")
                else:
                    logger.warning(f"Unexpected message type from {client_address}: {type(message)}")

            except asyncio.TimeoutError: # Silence detected
                if audio_bytes_buffer:
                    logger.info(f"Silence detected for {client_address}. Processing buffer ({len(audio_bytes_buffer)} bytes).")
                    await _transcribe_and_send_audio(audio_bytes_buffer, websocket, client_address)
                    audio_bytes_buffer = bytearray() # Reset for the next turn
                # If buffer is empty, timeout means continued silence, just loop again.

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"STT client {client_address} disconnected gracefully.")
        if audio_bytes_buffer:
            logger.info(f"Processing remaining {len(audio_bytes_buffer)} audio bytes from {client_address} after graceful disconnect.")
            await _transcribe_and_send_audio(audio_bytes_buffer, websocket, client_address)
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"STT client {client_address} connection closed with error: {e}")
        if audio_bytes_buffer:
            logger.info(f"Processing remaining {len(audio_bytes_buffer)} audio bytes from {client_address} after connection error.")
            await _transcribe_and_send_audio(audio_bytes_buffer, websocket, client_address)
    except Exception as e:
        logger.error(f"Error in STT handler for {client_address}: {e}", exc_info=True)
        try:
            # Attempt to inform client of server-side error
            # Use str(e) for safe serialization if e is a complex exception object
            await websocket.send(json.dumps({"transcription": "", "error": f"Server error: {str(e)}"}))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed for {client_address} before sending server error message.")
        except Exception as send_e: # Catch other errors during this specific send attempt
            logger.error(f"Failed to send error message to client {client_address}: {send_e}")
    finally:
        logger.info(f"Cleaning up for STT client {client_address}.")
        # Buffer should be cleared by processing logic. This is a final check.
        if audio_bytes_buffer:
            logger.warning(f"Final cleanup: Non-empty audio buffer ({len(audio_bytes_buffer)} bytes) for {client_address} being discarded.")
            audio_bytes_buffer = bytearray() # Explicitly clear


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
