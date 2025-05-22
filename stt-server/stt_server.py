import asyncio
import whisper
import numpy as np
import torch
import logging
import os
import json
import functools
from dotenv import load_dotenv
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
import torchaudio.transforms as T

# Load environment variables from .env file
load_dotenv()

# Configure logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Suppress aiortc's verbose DEBUG logs unless specifically needed
logging.getLogger("aiortc").setLevel(logging.INFO)
logging.getLogger("aioice").setLevel(logging.INFO) # aioice is used by aiortc

# STT Server Configuration
HOST = os.environ.get('STT_HOST', '0.0.0.0')  # Listen on all interfaces by default
PORT = int(os.environ.get('STT_PORT', 8766))  # Match client's expected port
MODEL_NAME = os.environ.get('MODEL_NAME', "tiny.en") # e.g., "tiny.en", "base", "small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SAMPLE_RATE = 16000 # Should match client's audio sample rate
WHISPER_BEAM_SIZE = int(os.environ.get('WHISPER_BEAM_SIZE', 5)) # Added beam_size configuration

# Load Whisper model (once at startup)
model = None
logger.info(f"Attempting to load Whisper model: {MODEL_NAME} on {DEVICE}...")
try:
    model = whisper.load_model(MODEL_NAME, device=DEVICE)
    logger.info(f"Whisper model '{MODEL_NAME}' loaded successfully on {DEVICE}.")
except Exception as e:
    logger.error(f"Failed to load Whisper model '{MODEL_NAME}': {e}", exc_info=True)
    logger.error("STT server will not be able to process audio.")

# --- WebRTC Configuration ---
WEBRTC_STT_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=[
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
            "stun:stun3.l.google.com:19302",
            "stun:stun4.l.google.com:19302"
        ])
    ]
)

class WebRTCSTTConnection:
    def __init__(self, pc_config, on_close_callback, whisper_model, stt_params, device):
        self.pc = RTCPeerConnection(configuration=pc_config)
        self.on_close_callback = on_close_callback
        self.whisper_model = whisper_model
        self.stt_params = stt_params
        self.device = device
        
        self.audio_track_receiver = None
        self.transcription_channel = None
        self.audio_buffer = bytearray()
        self.audio_processing_task = None
        self.closed = asyncio.Event()
        self.remote_address = "N/A"

        self._setup_pc_handlers()
        self._setup_transcription_channel()

    def _log_prefix(self):
        return f"STTConn-{self.remote_address}-{id(self)%1000}"

    def _setup_transcription_channel(self):
        self.transcription_channel = self.pc.createDataChannel("stt_transcriptions")
        logger.info(f"[{self._log_prefix()}] Data channel 'stt_transcriptions' created. Initial state: {self.transcription_channel.readyState}")

        @self.transcription_channel.on("open")
        async def on_open():
            logger.info(f"[{self._log_prefix()}] Data channel 'stt_transcriptions' opened. Current state: {self.transcription_channel.readyState}")

        @self.transcription_channel.on("close")
        async def on_close():
            logger.info(f"[{self._log_prefix()}] Data channel 'stt_transcriptions' closed. Current state: {self.transcription_channel.readyState}")

        @self.transcription_channel.on("error")
        async def on_error(error):
            logger.error(f"[{self._log_prefix()}] Data channel 'stt_transcriptions' error: {error}. Current state: {self.transcription_channel.readyState}")

    def _setup_pc_handlers(self):
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"[{self._log_prefix()}] Track {track.kind} received.")
            if track.kind == "audio":
                self.audio_track_receiver = track
                if not self.audio_processing_task or self.audio_processing_task.done():
                    # Only start processing if connection is already established,
                    # or it will be started by the 'connected' state change handler.
                    if self.pc.connectionState == "connected":
                        logger.info(f"[{self._log_prefix()}] Audio track received on already connected peer. Starting audio processing.")
                        self.audio_processing_task = asyncio.create_task(self._process_incoming_audio())
                    else:
                        logger.info(f"[{self._log_prefix()}] Audio track received, but PC not yet connected. Deferring audio processing start.")
            # No @track.on("ended") here; _process_incoming_audio handles track end via recv() exception

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"[{self._log_prefix()}] Data channel '{channel.label}' offered by remote (STT server expects to create its own).")
            # STT server primarily sends on its own created channel.
            # Could listen here for control messages if protocol evolves.

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            # This event fires when a local ICE candidate is gathered.
            # aiortc handles sending these as part of the offer/answer exchange unless trickle ICE is manually implemented.
            logger.info(f"[{self._log_prefix()}] Local ICE candidate gathered: {candidate}")

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"[{self._log_prefix()}] PC Connection state: {self.pc.connectionState}")
            if self.pc.connectionState in {"failed", "closed", "disconnected"}:
                await self.close()
            elif self.pc.connectionState == "connected":
                logger.info(f"[{self._log_prefix()}] PC Connection established successfully.")
                if self.audio_track_receiver and (not self.audio_processing_task or self.audio_processing_task.done()):
                     logger.info(f"[{self._log_prefix()}] Initiating audio processing on 'connected' state.")
                     self.audio_processing_task = asyncio.create_task(self._process_incoming_audio())

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"[{self._log_prefix()}] ICE connection state: {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState in {"failed", "closed", "disconnected"}:
                logger.warning(f"[{self._log_prefix()}] ICE connection failed or closed. Closing PC.")
                await self.close() # Attempt to close if ICE fails

    async def _process_incoming_audio(self):
        logger.info(f"[{self._log_prefix()}] Starting audio processing.")
        self.audio_buffer = bytearray()
        resampler = None
        target_sr = self.stt_params.get("target_sample_rate", TARGET_SAMPLE_RATE)

        if not self.audio_track_receiver:
            logger.error(f"[{self._log_prefix()}] No audio track receiver set.")
            return

        try:
            while not self.closed.is_set():
                try:
                    frame = await self.audio_track_receiver.recv()
                except Exception: # MediaStreamError indicates track ended or closed
                    logger.info(f"[{self._log_prefix()}] Audio track ended or connection closed.")
                    break
                
                # Correctly access audio data from av.AudioFrame
                audio_bytes = bytes(frame.planes[0])
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                if frame.layout.nb_channels > 1:
                    audio_int16 = audio_int16.reshape(-1, frame.layout.nb_channels).mean(axis=1).astype(np.int16)
                
                audio_tensor = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)

                if frame.sample_rate != target_sr:
                    if resampler is None or getattr(resampler, 'orig_freq', 0) != frame.sample_rate:
                        logger.info(f"[{self._log_prefix()}] Initializing resampler: {frame.sample_rate}Hz -> {target_sr}Hz")
                        resampler = T.Resample(orig_freq=frame.sample_rate, new_freq=target_sr)
                    audio_tensor = resampler(audio_tensor)
                
                processed_audio_int16 = (audio_tensor.cpu().numpy() * 32768.0).astype(np.int16)
                self.audio_buffer.extend(processed_audio_int16.tobytes())
        except asyncio.CancelledError:
            logger.info(f"[{self._log_prefix()}] Audio processing task cancelled.")
        except Exception as e:
            logger.error(f"[{self._log_prefix()}] Error in audio processing: {e}", exc_info=True)
        finally:
            logger.info(f"[{self._log_prefix()}] Audio processing finished. Buffer: {len(self.audio_buffer)} bytes.")
            if self.audio_buffer:
                await self._transcribe_and_send_buffer()
            else: # Send empty if no audio but track ended, to signal completion if needed
                await self._send_transcription_result("", "No audio data received on track.")
            self.audio_processing_task = None

    async def _transcribe_and_send_buffer(self):
        if not self.whisper_model:
            logger.error(f"[{self._log_prefix()}] Whisper model not loaded.")
            await self._send_transcription_error("Whisper model not loaded on server.")
            return

        if not self.audio_buffer:
            logger.warning(f"[{self._log_prefix()}] Empty audio buffer for transcription.")
            await self._send_transcription_result("", "Empty audio buffer for transcription")
            return

        audio_data_bytes = bytes(self.audio_buffer)
        self.audio_buffer.clear()

        bytes_in_seconds = len(audio_data_bytes) / 2 / TARGET_SAMPLE_RATE  # 16-bit audio = 2 bytes per sample
        logger.info(f"[{self._log_prefix()}] Transcribing {len(audio_data_bytes)} bytes ({bytes_in_seconds:.2f} seconds).")
        try:
            audio_int16 = np.frombuffer(audio_data_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"[{self._log_prefix()}] Audio conversion error: {e}")
            await self._send_transcription_error(f"Audio conversion error: {e}")
            return

        transcription_text, transcription_error = "", None
        try:
            if audio_float32.ndim > 1: audio_float32 = audio_float32.flatten()
            if audio_float32.size == 0:
                await self._send_transcription_error("Processed audio data is empty")
                return

            use_fp16 = self.device == "cuda"
            loop = asyncio.get_running_loop()
            transcribe_options = {"fp16": use_fp16, "beam_size": self.stt_params.get('beam_size')}
            if not transcribe_options["beam_size"] or transcribe_options["beam_size"] <=0: # Whisper uses default if None or 0
                del transcribe_options["beam_size"]

            logger.info(f"[{self._log_prefix()}] Running Whisper transcription with options: {transcribe_options}")
            transcribe_func = functools.partial(self.whisper_model.transcribe, audio=audio_float32, **transcribe_options)
            result = await loop.run_in_executor(None, transcribe_func)
            transcription_text = result["text"].strip()  # Trim whitespace
            logger.info(f"[{self._log_prefix()}] Raw Whisper result: {result}")
            logger.info(f"[{self._log_prefix()}] Transcription: {transcription_text}")
        except Exception as e:
            logger.error(f"[{self._log_prefix()}] Transcription error: {e}", exc_info=True)
            transcription_error = f"Transcription error: {str(e)}"
        
        await self._send_transcription_result(transcription_text, transcription_error)

    async def _send_transcription_result(self, text, error_msg=None):
        if self.transcription_channel and self.transcription_channel.readyState == "open":
            payload = {
                "type": "stt_transcription",
                "data": {
                    "transcription": text,
                    "error": error_msg
                }
            }
            try:
                self.transcription_channel.send(json.dumps(payload))
                logger.info(f"[{self._log_prefix()}] Sent transcription: {text[:100]}{'...' if len(text) > 100 else ''}")
            except Exception as e:
                logger.error(f"[{self._log_prefix()}] Failed to send transcription via data channel: {e}")
        else:
            logger.warning(f"[{self._log_prefix()}] Transcription channel not open. Cannot send: T='{text}', E='{error_msg}'")

    async def _send_transcription_error(self, error_msg):
        await self._send_transcription_result("", error_msg)

    async def close(self):
        if not self.closed.is_set():
            self.closed.set()
            logger.info(f"[{self._log_prefix()}] Closing WebRTCSTTConnection.")
            if self.audio_processing_task:
                self.audio_processing_task.cancel()
            if self.transcription_channel and self.transcription_channel.readyState == "open":
                self.transcription_channel.close()
            if self.pc.signalingState != "closed": await self.pc.close()
            if self.on_close_callback: self.on_close_callback(self)
            logger.info(f"[{self._log_prefix()}] WebRTCSTTConnection closed.")

class STTServer:
    def __init__(self, host, port, whisper_model, device):
        self.host, self.port = host, port
        self.whisper_model, self.device = whisper_model, device
        self.connections = set()
        self.stt_params = {"target_sample_rate": TARGET_SAMPLE_RATE, "beam_size": WHISPER_BEAM_SIZE}

    def _remove_connection(self, conn):
        logger.info(f"Removing STT connection: {conn._log_prefix()}")
        self.connections.discard(conn)

    async def handle_options_stt(self, request): # Basic CORS preflight
        return web.Response(status=200, headers={
            'Access-Control-Allow-Origin': 'http://localhost:3000',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
        })

    async def handle_offer_stt(self, request):
        if not self.whisper_model:
            return web.Response(
                status=503,
                text="STT service unavailable: Model not loaded.",
                headers={'Access-Control-Allow-Origin': 'http://localhost:3000'}
            )
        try:
            params = await request.json()
            if not params.get("sdp") or params.get("type") != "offer":
                return web.Response(
                    status=400,
                    text="Invalid WebRTC offer.",
                    headers={'Access-Control-Allow-Origin': 'http://localhost:3000'}
                )

            conn = WebRTCSTTConnection(WEBRTC_STT_CONFIG, self._remove_connection, 
                                       self.whisper_model, self.stt_params, self.device)
            conn.remote_address = str(request.remote)
            
            conn.pc.addTransceiver("audio", direction="recvonly") # Expect to receive audio
            # Transcription channel is created by WebRTCSTTConnection constructor

            await conn.pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
            answer = await conn.pc.createAnswer()
            await conn.pc.setLocalDescription(answer)
            
            self.connections.add(conn)
            logger.info(f"STT connection {conn._log_prefix()} added. Total: {len(self.connections)}")
            return web.json_response(
                {"sdp": answer.sdp, "type": answer.type},
                headers={
                    'Access-Control-Allow-Origin': 'http://localhost:3000',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                }
            )
        except Exception as e:
            logger.exception(f"Error handling STT offer: {e}")
            if 'conn' in locals() and conn: await conn.close()
            return web.Response(
                status=500,
                text="Internal server error processing STT offer.",
                headers={'Access-Control-Allow-Origin': 'http://localhost:3000'}
            )

    async def start(self):
        if not self.whisper_model:
            logger.error("Whisper model not loaded. STT server cannot start.")
            return

        app = web.Application()
        app.router.add_post('/offer_stt', self.handle_offer_stt)
        app.router.add_route('OPTIONS', '/offer_stt', self.handle_options_stt)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"STT HTTP/WebRTC server started on http://{self.host}:{self.port}/offer_stt")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt: logger.info("STT server shutting down...")
        finally:
            logger.info("Cleaning up STT server resources...")
            for conn_task in [c.close() for c in list(self.connections)]: await conn_task
            await runner.cleanup()
            logger.info("STT server stopped.")

if __name__ == "__main__":
    if not model:
        logger.critical("Whisper model failed to load. Exiting.")
        exit(1)
        
    server = STTServer(host=HOST, port=PORT, whisper_model=model, device=DEVICE)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("STT server shutting down...")
    except Exception as e:
        logger.critical(f"STT server failed catastrophically: {e}", exc_info=True)
