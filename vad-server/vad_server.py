# /root/silero-vad-test/vad_server.py
import asyncio
import torch
import os
import logging
import json
import sys
from dotenv import load_dotenv
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer, AudioStreamTrack, MediaStreamError
from aiohttp import web
from aiohttp.web import middleware # If middleware is still used
from aiohttp import ClientSession as AiohttpClientSession

import numpy as np
import torchaudio
import torchaudio.transforms as T
import aiohttp # For making HTTP requests to STT server
from aiortc.contrib.media import AudioFrame # For sending audio to STT
# Load environment variables from .env file if it exists
load_dotenv() 

# CORS middleware
@middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)
    
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# --- Logging Setup ---
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Suppress aiortc's verbose DEBUG logs unless specifically needed
logging.getLogger("aiortc").setLevel(logging.INFO)
logging.getLogger("aioice").setLevel(logging.INFO) # aioice is used by aiortc


# --- Silero VAD Setup ---
VAD_MODEL = None
VAD_ITERATOR_CLASS = None
TARGET_SAMPLE_RATE = 16000  # Silero VAD expects 16kHz

# --- VAD Configuration (from environment or defaults) ---
VAD_THRESHOLD = float(os.getenv('VAD_THRESHOLD', 0.6)) 
VAD_MIN_SILENCE_DURATION_MS = int(os.getenv('VAD_MIN_SILENCE_DURATION_MS', 300)) # Increased from 100ms
VAD_SPEECH_PAD_MS = int(os.getenv('VAD_SPEECH_PAD_MS', 200)) # Increased default from 30ms
VAD_MIN_SPEECH_DURATION_MS = int(os.getenv('VAD_MIN_SPEECH_DURATION_MS', 250))

# VAD Server Configuration
HOST = os.environ.get('HOST', 'localhost')
PORT = int(os.environ.get('PORT', 8765))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 512  # Silero VAD expects 512 samples for 16kHz audio (32ms)
SAMPLING_RATE = 16000

# STT Server Configuration (for VAD server to connect to)
STT_SERVER_OFFER_URL = os.getenv("STT_SERVER_OFFER_URL", f"http://{os.getenv('STT_HOST', 'localhost')}:{os.getenv('STT_PORT', 8768)}/offer_stt")
logger.info(f"VAD Server configured to connect to STT at: {STT_SERVER_OFFER_URL}")

# WebRTC Configuration
WEBRTC_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls="stun:stun.l.google.com:19302")
        # You can add more RTCIceServer objects here if needed
    ]
)

WEBRTC_CONNECTION_TIMEOUT = 10  # seconds

def load_silero_vad_model():
    global VAD_MODEL, VAD_ITERATOR_CLASS
    try:
        logger.info("Attempting to load Silero VAD model using torch.hub.load...")
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False, # Set to True to always re-download
                                      trust_repo=True)
        VAD_MODEL = model.to(DEVICE) # Move the model to the configured device
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils
        VAD_ITERATOR_CLASS = VADIterator
        logger.info(f"Silero VAD model loaded and moved to {DEVICE} successfully.")
    except Exception as e:
        logger.exception(f"Error during Silero VAD setup: {e}")
        sys.exit(1)

class STTUtteranceProcessor:
    """Handles a one-shot WebRTC connection to the STT server for a single utterance."""
    def __init__(self, stt_server_url, audio_tensor_float32_mono_16khz, on_complete_callback, webrtc_config):
        self.stt_server_url = stt_server_url
        self.audio_tensor = audio_tensor_float32_mono_16khz
        self.on_complete_callback = on_complete_callback # Called with (transcription_json or error_info)
        self.pc = RTCPeerConnection(configuration=webrtc_config)
        self.audio_sender_track = None
        self.transcription_channel = None
        self._done_event = asyncio.Event()
        self._setup_pc_handlers()

    def _setup_pc_handlers(self):
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"STTProcessor ICE state: {self.pc.iceConnectionState}") # Changed to INFO
            if self.pc.iceConnectionState in {"failed", "closed", "disconnected"}:
                if not self._done_event.is_set():
                    logger.warning(f"STTProcessor ICE failed/closed before completion.")
                    await self.on_complete_callback({"transcription": "", "error": "STT connection failed (ICE)"})
                    self._done_event.set() # Ensure run() loop can exit
                await self.close()

        @self.pc.on("connectionstatechange") # Add this for overall PC state
        async def on_connectionstatechange():
            logger.info(f"STTProcessor PC Connection State: {self.pc.connectionState}")
            if self.pc.connectionState in {"failed", "closed", "disconnected"}:
                if not self._done_event.is_set(): # Ensure callback if not already done
                    logger.warning(f"STTProcessor PC connection failed/closed before completion.")
                    await self.on_complete_callback({"transcription": "", "error": "STT PC connection failed"})
                    self._done_event.set()
                await self.close()

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            # This event fires when a local ICE candidate is gathered by the STTProcessor.
            # aiortc handles sending these as part of the offer/answer exchange.
            logger.info(f"STTProcessor: Local ICE candidate gathered: {candidate}")

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label == "stt_transcriptions":
                logger.info(f"STTProcessor: Data channel '{channel.label}' received from STT server.")
                self.transcription_channel = channel
                @channel.on("message")
                async def on_message(message):
                    logger.info(f"STTProcessor: Transcription received: {message[:100]}...")
                    try:
                        data = json.loads(message)
                        await self.on_complete_callback(data)
                    except json.JSONDecodeError:
                        logger.error(f"STTProcessor: Failed to parse STT JSON: {message}")
                        await self.on_complete_callback({"transcription": "", "error": "Invalid STT response format"})
                    finally:
                        self._done_event.set() # Signal completion
                @channel.on("open")
                async def on_open():
                    logger.info(f"STTProcessor: Data channel '{channel.label}' is open and ready to receive messages.")
            else:
                logger.warning(f"STTProcessor: Unexpected data channel '{channel.label}', closing it.")
                channel.close()

    async def run(self):
        try:
            # Create a custom AudioStreamTrack that will send our audio data
            class AudioSenderTrack(MediaStreamTrack):
                kind = "audio"

                def __init__(self, audio_tensor):
                    super().__init__()
                    self.audio_tensor = audio_tensor
                    self.samples_per_frame = int(TARGET_SAMPLE_RATE * 0.02)  # 20ms frames
                    self.current_frame_idx = 0
                    self.frames = []
                    
                    # Convert audio to frames upfront
                    audio_int16_numpy = (self.audio_tensor.cpu().numpy() * 32767).astype(np.int16)
                    idx = 0
                    while idx < len(audio_int16_numpy):
                        end_sample = min(idx + self.samples_per_frame, len(audio_int16_numpy))
                        frame_np = audio_int16_numpy[idx:end_sample].reshape(1, -1)
                        audio_frame = AudioFrame.from_ndarray(frame_np, format='s16', layout='mono')
                        audio_frame.sample_rate = TARGET_SAMPLE_RATE
                        self.frames.append(audio_frame)
                        idx = end_sample
                    logger.info(f"AudioSenderTrack: Prepared {len(self.frames)} frames")

                async def recv(self):
                    if self.current_frame_idx < len(self.frames):
                        frame = self.frames[self.current_frame_idx]
                        self.current_frame_idx += 1
                        return frame
                    else:
                        self.stop()
                        raise MediaStreamError("End of audio track")

            audio_track = AudioSenderTrack(self.audio_tensor)
            self.pc.addTrack(audio_track)

            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            payload = {"sdp": offer.sdp, "type": offer.type}
            logger.info(f"STTProcessor: Attempting to POST offer to STT server at {self.stt_server_url}")
            async with AiohttpClientSession() as session:
                try:
                    async with session.post(self.stt_server_url, json=payload, timeout=10) as response:
                        logger.info(f"STTProcessor: Received response status {response.status} from STT server.")
                        response.raise_for_status()
                        answer_data = await response.json()
                        logger.info(f"STTProcessor: Received answer from STT server (first 200 chars): {json.dumps(answer_data)[:200]}")
                        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"]))
                except aiohttp.ClientError as e:
                    logger.error(f"STTProcessor: HTTP ClientError connecting to STT server at {self.stt_server_url}: {e}", exc_info=True)
                    await self.on_complete_callback({"transcription": "", "error": f"STT connection error: {str(e)}"})
                    self._done_event.set()
                    return
                except json.JSONDecodeError as e:
                    logger.error(f"STTProcessor: Failed to decode JSON answer from STT server: {e}", exc_info=True)
                    raw_text = await response.text()
                    logger.error(f"STTProcessor: Raw response text from STT server (first 500 chars): {raw_text[:500]}")
                    await self.on_complete_callback({"transcription": "", "error": "Invalid JSON answer from STT server"})
                    self._done_event.set()
                    return

            logger.info(f"STTProcessor: Successfully set remote description from STT server's answer.")
            logger.info(f"STTProcessor: Connected to STT server. Audio track added ({self.audio_tensor.shape[0]/TARGET_SAMPLE_RATE:.2f}s).")
            
            # The audio will be sent automatically through the WebRTC connection as the remote peer
            # calls recv() on their end. We just need to wait for the completion signal.

            # Signal end of audio by stopping the track (aiortc specific way)
            # This will trigger "track ended" on the STT server side.
            # A more robust way is to close the sender or the whole PC after getting the response.
            # For now, STT server processes when its receiver track ends. This happens when PC closes.

            await asyncio.wait_for(self._done_event.wait(), timeout=60.0) # Wait for transcription
        except asyncio.TimeoutError:
            logger.warning(f"STTProcessor: Timeout waiting for transcription or connection.")
            await self.on_complete_callback({"transcription": "", "error": "Timeout with STT service"})
        except Exception as e:
            logger.exception(f"STTProcessor: Error during STT processing: {e}")
            await self.on_complete_callback({"transcription": "", "error": f"STT client error: {str(e)}"})
        finally:
            await self.close()

    async def close(self):
        logger.info("STTProcessor: Closing connection.")
        # self.audio_sender_track is no longer an instance variable of STTUtteranceProcessor
        # The track is managed by the RTCPeerConnection and its senders.
        pass
        if self.pc and self.pc.signalingState != "closed":
            await self.pc.close()

class WebRTCConnection:
    def __init__(self, config, on_close_callback, vad_model, vad_iterator_class, vad_params, device):
        self.pc = RTCPeerConnection(configuration=config)
        self.data_channel_to_client = None # Channel server uses to send VAD status
        self.audio_track = None
        self.connected = asyncio.Event()
        self.closed = asyncio.Event()
        self.vad_status = False
        self.on_close_callback = on_close_callback
        self.audio_processing_task = None
        self.vad_model, self.vad_iterator_class, self.vad_params, self.device = vad_model, vad_iterator_class, vad_params, device
        self.tensor_buffer = torch.empty(0, device=self.device)
        self.speech_buffer = torch.empty(0, device=self.device) # Buffer for current speech segment
        self.is_currently_speech = False
        self._setup_connection_handlers()

    def _setup_connection_handlers(self):
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if self.pc.connectionState == "failed":
                await self.close()
            elif self.pc.connectionState == "connected":
                logger.info("WebRTC connection established.")
                self.connected.set()

        @self.pc.on("datachannel")
        def on_datachannel_from_client(channel): # Handles channels opened by the client
            logger.info(f"Data channel '{channel.label}' received from client.")
            @channel.on("message")
            async def on_message(message):
                logger.info(f"Message from client on '{channel.label}': {message}")
                if message == "get_vad_status":
                    if self.data_channel_to_client and self.data_channel_to_client.readyState == "open":
                        await self.data_channel_to_client.send(json.dumps({"vad_status": self.vad_status}))
                    elif channel.readyState == "open": # Fallback to sending on the requesting channel
                         await channel.send(json.dumps({"vad_status": self.vad_status}))

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                logger.info(f"Audio track {track.id} received (Sample Rate: {track.kind} Hz, Channels: {getattr(track, 'channels', 'N/A')})")
                self.audio_track = track
                asyncio.create_task(self.start_audio_processing())

                @track.on("ended") # type: ignore
                async def on_ended():
                    logger.info(f"Audio track {track.id} ended.")
                    await self.close()

    async def start_audio_processing(self):
        if self.audio_track and not self.audio_processing_task:
            logger.info("Starting audio processing task...")
            self.audio_processing_task = asyncio.create_task(self._process_audio_track())
            try:
                await self.audio_processing_task
            except asyncio.CancelledError:
                logger.info("Audio processing task cancelled.")
            except Exception as e:
                logger.exception(f"Audio processing task failed: {e}")
            finally:
                self.audio_processing_task = None
                logger.info("Audio processing task finished.")

    async def _process_audio_track(self):
        logger.info(f"[{id(self)}] ENTERING _process_audio_track. Application LOG_LEVEL is set to: {LOG_LEVEL}")
        vad_iterator = self.vad_iterator_class(
            self.vad_model,
            threshold=self.vad_params['threshold'],
            sampling_rate=TARGET_SAMPLE_RATE, # We will resample to this
            min_silence_duration_ms=self.vad_params['min_silence_duration_ms'],
            speech_pad_ms=self.vad_params['speech_pad_ms']
        )
        resampler = None
        min_speech_samples = int(TARGET_SAMPLE_RATE * self.vad_params['min_speech_duration_ms'] / 1000)

        try:
            while not self.closed.is_set() and self.audio_track:
                # logger.debug(f"[{id(self)}] _process_audio_track: Top of main while loop. Waiting for audio frame...")
                frame = await self.audio_track.recv()
                if self.closed.is_set(): break
                # Correctly access audio data from av.AudioFrame
                audio_bytes = bytes(frame.planes[0])
                # logger.debug(f"[{id(self)}] _process_audio_track: Received audio frame (size: {len(audio_bytes)} bytes, SR: {frame.sample_rate}, Chan: {frame.layout.nb_channels}, Format: {frame.format.name}).")
                raw_audio = np.frombuffer(audio_bytes, dtype=np.int16)


                if frame.layout.nb_channels > 1:
                    raw_audio = raw_audio.reshape(-1, frame.layout.nb_channels).mean(axis=1).astype(np.int16)

                audio_tensor = torch.from_numpy(raw_audio.astype(np.float32) / 32768.0).to(self.device)

                if frame.sample_rate != TARGET_SAMPLE_RATE:
                    if resampler is None or getattr(resampler, 'orig_freq', 0) != frame.sample_rate:
                        logger.info(f"Initializing resampler from {frame.sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz")
                        resampler = T.Resample(
                            orig_freq=frame.sample_rate,
                            new_freq=TARGET_SAMPLE_RATE
                        ).to(self.device)
                    audio_tensor = resampler(audio_tensor)
                
                self.tensor_buffer = torch.cat((self.tensor_buffer, audio_tensor))

                while self.tensor_buffer.shape[0] >= CHUNK_SIZE:
                    # logger.debug(f"[{id(self)}] _process_audio_track: Tensor buffer has enough data ({self.tensor_buffer.shape[0]} samples). Processing chunk of {CHUNK_SIZE} samples.")
                    current_tensor_chunk = self.tensor_buffer[:CHUNK_SIZE]
                    self.tensor_buffer = self.tensor_buffer[CHUNK_SIZE:]

                    # Process with VADIterator
                    speech_dict = vad_iterator(current_tensor_chunk, return_seconds=False) 
                    # vad_iterator may internally buffer and yield speech chunks when ready.
                    # logger.debug(f"[{id(self)}] _process_audio_track: Post VAD iteration. vad_iterator.triggered = {vad_iterator.triggered}, self.is_currently_speech (local state) = {self.is_currently_speech}")
                    # The VADIterator used here seems to update `current_speech` state.
                    # We need to collect audio when `current_speech` is true.

                    if vad_iterator.triggered and not self.is_currently_speech: # Speech started
                        self.is_currently_speech = True
                        # Clear previous buffer if any, though it should be empty here
                        self.speech_buffer = torch.empty(0, device=self.device)
                        self.speech_buffer = current_tensor_chunk # Start with current chunk                        
                        # logger.debug(f"VAD: Speech segment started. Initial chunk buffered ({current_tensor_chunk.shape[0]} samples).")
                    elif vad_iterator.triggered and self.is_currently_speech: # Speech continues
                        self.speech_buffer = torch.cat((self.speech_buffer, current_tensor_chunk))
                        # logger.debug(f"VAD: Speech continues. Buffer size: {self.speech_buffer.shape[0]} samples.")
                    elif not vad_iterator.triggered and self.is_currently_speech: # Speech ended
                        self.is_currently_speech = False
                        logger.info(f"VAD: Speech segment ended. Buffer size: {self.speech_buffer.shape[0]} samples. Min required: {min_speech_samples} samples.")
                        if self.speech_buffer.shape[0] >= min_speech_samples:
                            logger.info(f"VAD: CONDITION MET: Speech segment long enough. Preparing to send to STT server at {STT_SERVER_OFFER_URL}.")
                            logger.info(f"VAD: Sending speech segment ({self.speech_buffer.shape[0]/TARGET_SAMPLE_RATE:.2f}s) to STT.")
                            # Create a task to process this utterance with STT server
                            stt_processor = STTUtteranceProcessor(
                                STT_SERVER_OFFER_URL, 
                                self.speech_buffer.clone(), # Send a clone
                                self._handle_stt_result,
                                WEBRTC_CONFIG # Use same STUN config
                            )
                            asyncio.create_task(stt_processor.run())
                        else:
                            logger.info(f"VAD: Speech segment too short ({self.speech_buffer.shape[0]} samples vs {min_speech_samples} required), discarding.")
                        self.speech_buffer = torch.empty(0, device=self.device) # Reset speech buffer

                    # Update VAD status for client
                    if vad_iterator.triggered != self.vad_status: # Compare with vad_iterator.triggered
                        self.vad_status = vad_iterator.triggered # Update self.vad_status based on vad_iterator.triggered
                        if self.data_channel_to_client and self.data_channel_to_client.readyState == "open":
                            logger.debug(f"Sending VAD status: {self.vad_status}")
                            await self.data_channel_to_client.send(json.dumps({"vad_status": self.vad_status}))
        except MediaStreamError:
            # This is an expected error when the client's audio track ends.
            logger.info(f"[{id(self)}] Audio track ended (MediaStreamError caught). Normal termination of audio processing for this track.")
        except Exception as e:
            # Log other unexpected errors.
            logger.exception(f"Error in audio processing loop: {e}")
        finally:
            logger.info("Exiting audio processing loop.")

    async def close(self):
        if not self.closed.is_set():
            self.closed.set()
            logger.info("Closing WebRTCConnection.")
            if self.audio_processing_task:
                self.audio_processing_task.cancel()
            if self.pc.signalingState != "closed":
                await self.pc.close()
            if self.on_close_callback:
                self.on_close_callback(self)

    async def _handle_stt_result(self, transcription_json):
        """Callback for STTUtteranceProcessor, relays result to browser client."""
        if self.data_channel_to_client and self.data_channel_to_client.readyState == "open":
            logger.info(f"Relaying STT result to browser: {transcription_json}")
            await self.data_channel_to_client.send(
                json.dumps({"type": "stt_transcription", "data": transcription_json})
            )
        else:
            logger.warning("Cannot relay STT result to browser: data channel not open or available.")

class VADServer:
    def __init__(self):
        self.webrtc_connections = set()

    def _remove_webrtc_connection(self, conn):
        self.webrtc_connections.discard(conn)
        logger.info(f"WebRTC connection removed. Total connections: {len(self.webrtc_connections)}")

    async def handle_webrtc_offer(self, request):
        if request.method == "OPTIONS":
            return web.Response()

        conn = None
        try:
            data = await request.json()
            logger.info(f"Received WebRTC offer payload: {json.dumps(data)}") # Log the entire payload

            received_sdp = data.get("sdp")
            received_type = data.get("type")

            if not received_sdp:
                error_msg = "Missing 'sdp' in WebRTC offer from client."
                logger.error(error_msg)
                return web.Response(status=400, text=error_msg)
            if not received_type:
                error_msg = "Missing 'type' in WebRTC offer from client."
                logger.error(error_msg)
                return web.Response(status=400, text=error_msg)
            if received_type.lower() != "offer": # Basic check for offer type
                error_msg = f"Expected WebRTC offer type 'offer', but got '{received_type}'."
                logger.error(error_msg)
                return web.Response(status=400, text=error_msg)
            
            vad_processing_params = {
                'threshold': VAD_THRESHOLD,
                'min_silence_duration_ms': VAD_MIN_SILENCE_DURATION_MS,
                'speech_pad_ms': VAD_SPEECH_PAD_MS,
                'min_speech_duration_ms': VAD_MIN_SPEECH_DURATION_MS # This param is for logic, not VADIterator init
            }
            if VAD_MODEL is None or VAD_ITERATOR_CLASS is None:
                error_msg = "VAD model or VAD iterator not loaded on server. Cannot process offer."
                logger.error(error_msg)
                return web.Response(status=500, text=error_msg)

            conn = WebRTCConnection(
                    config=WEBRTC_CONFIG, # Make sure this is the RTCConfiguration object
                    on_close_callback=self._remove_webrtc_connection,
                    vad_model=VAD_MODEL,
                    vad_iterator_class=VAD_ITERATOR_CLASS,
                    vad_params=vad_processing_params,
                    device=DEVICE
                )
            self.webrtc_connections.add(conn)
            
            # Create answer for the offer
            remote_offer_desc = RTCSessionDescription(
                sdp=received_sdp,
                type=received_type
            )
            logger.info(f"Attempting to set remote description with offer: type={remote_offer_desc.type}, sdp_len={len(remote_offer_desc.sdp) if remote_offer_desc.sdp else 'None'}")
            await conn.pc.setRemoteDescription(remote_offer_desc)
            logger.info("Successfully set remote description.")
            logger.info("Transceivers after setRemoteDescription:")
            for t_idx, t in enumerate(conn.pc.getTransceivers()):
                # _offerDirection is internal, use getattr for safety
                offer_dir = getattr(t, '_offerDirection', 'N/A')
                logger.info(f"  T{t_idx}: mid={t.mid}, kind={t.kind}, direction={t.direction}, _offerDirection={offer_dir}, currentDirection={t.currentDirection}")
            
            # By removing the explicit addTransceiver for audio, we rely on setRemoteDescription
            # having processed the client's audio offer, and createAnswer to formulate
            # the appropriate response for receiving that audio.
            logger.info("Skipping explicit addTransceiver for audio, relying on offer processing.")
            
            # Create data channel for VAD status
            # Server creates this channel to send VAD status updates to the client
            logger.info("Creating data channel 'vad_status_feed'.")
            server_created_channel = conn.pc.createDataChannel("vad_status_feed")
            conn.data_channel_to_client = server_created_channel
            logger.info("Data channel 'vad_status_feed' created.")

            @server_created_channel.on("open")
            def on_open():
                logger.info(f"Data channel '{server_created_channel.label}' opened by server.")

            @server_created_channel.on("message") # Listen for any messages from client on this channel
            async def on_message(message): # e.g. client sends "ping"
                logger.info(f"Message from client on '{server_created_channel.label}': {message}")
            
            # Create and set local description
            logger.info("Attempting to create answer.")
            answer = await conn.pc.createAnswer()
            logger.info("Answer created.")
            logger.info(f"Attempting to set local description with answer: type={answer.type}, sdp_len={len(answer.sdp) if answer.sdp else 'None'}")
            await conn.pc.setLocalDescription(answer)
            logger.info("Successfully set local description (answer).")
            
            response_data = {
                "sdp": conn.pc.localDescription.sdp,
                "type": conn.pc.localDescription.type
            }
            logger.info(f"Sending WebRTC answer: {json.dumps(response_data)}")
            
            return web.json_response(response_data)

        except json.JSONDecodeError as je:
            logger.exception("Failed to decode JSON from request.")
            return web.Response(status=400, text=f"Invalid JSON payload: {str(je)}")
        except Exception as e:
            # Log the full traceback for any other exception
            logger.exception(f"Error handling WebRTC offer. Exception type: {type(e)}, message: {str(e)}")
            if conn: # If conn object was created, ensure it's closed
                await conn.close()
            # Return a more generic error to client, but server logs will have details
            return web.Response(
                status=500, # Using 500 as it's likely an internal server issue if SDP parsing fails unexpectedly
                text="Internal server error during WebRTC offer processing."
            )

    async def start(self):
        app = web.Application(middlewares=[cors_middleware])
        app.router.add_route('OPTIONS', '/offer', self.handle_webrtc_offer)
        app.router.add_post('/offer', self.handle_webrtc_offer)
        
        logger.info(f"VAD server starting. WebRTC signaling will be on port {PORT + 1}")
        
        runner = web.AppRunner(app)
        await runner.setup()
        # WebRTC HTTP signaling server runs on PORT + 1
        site = web.TCPSite(runner, HOST, PORT + 1) 
        await site.start()
        logger.info(f"WebRTC HTTP signaling server running on http://{HOST}:{PORT + 1}/offer")

        # Keep the aiohttp server running
        try:
            await asyncio.Event().wait() # Keep running indefinitely
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            await runner.cleanup()
            logger.info("Server stopped.")

if __name__ == "__main__":
    load_silero_vad_model()
    if VAD_MODEL is None:
        logger.error("Server cannot start: VAD model failed to load.")
        sys.exit(1)

    server = VADServer()
    asyncio.run(server.start())
