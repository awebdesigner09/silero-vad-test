// src/AudioVADClient.js (or wherever you place your React components)
import React, { useState, useEffect, useRef } from 'react';

const VAD_SERVER_URL = 'ws://localhost:8765'; // VAD server
const TARGET_SAMPLE_RATE = 16000; // Must match server and VAD model. Silero VAD expects 512 samples for 16kHz.
const SCRIPT_PROCESSOR_BUFFER_SIZE = 512; // Or 2048, 4096. Affects latency and chunk size. For 16kHz, Silero VAD expects 512.
const STT_SERVER_URL = 'ws://localhost:8766'; // NEW: STT Server URL

function AudioVADClient() {
    const [isRecording, setIsRecording] = useState(false);
    const [vadStatus, setVadStatus] = useState('Idle');
    const [serverStatus, setServerStatus] = useState('Disconnected');
    const isRecordingRef = useRef(isRecording);

    // NEW: STT related state and refs
    const [sttServerStatus, setSttServerStatus] = useState('Disconnected');
    const [transcription, setTranscription] = useState('');
    const sttSocketRef = useRef(null);
    const audioBufferForSttRef = useRef([]); // To store Int16Array chunks for current speech segment
    const isSpeechSegmentActiveRef = useRef(false); // True between VAD_SPEECH_START and VAD_SPEECH_END

    // NEW: Microphone selection state
    const [availableMics, setAvailableMics] = useState([]);
    const [selectedMicId, setSelectedMicId] = useState(''); // Empty string for default microphone

    const audioContextRef = useRef(null);
    const mediaStreamRef = useRef(null);
    const scriptProcessorRef = useRef(null);
    const socketRef = useRef(null);

    // Effect for WebSocket connection management
    useEffect(() => {
        isRecordingRef.current = isRecording;
    }, [isRecording]);


    useEffect(() => {
        // Connect to VAD server on mount
        connectVadWebSocket(); // Renamed for clarity

        return () => { // Cleanup on unmount
            console.log("AudioVADClient: useEffect cleanup running");
            if (isRecordingRef.current) { // Use ref to get current recording state
                stopRecordingLogic();
            }
            if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
                console.log("AudioVADClient: Closing WebSocket from useEffect cleanup");
                socketRef.current.close();
            }
            
            socketRef.current = null; // Nullify the ref
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Run once on mount
    
    // NEW: useEffect for STT WebSocket connection management
    useEffect(() => {
        connectSttWebSocket();

        return () => {
            console.log("AudioVADClient: STT useEffect cleanup running");
            if (sttSocketRef.current && sttSocketRef.current.readyState === WebSocket.OPEN) {
                console.log("AudioVADClient: Closing STT WebSocket from useEffect cleanup");
                sttSocketRef.current.close();
            }
            sttSocketRef.current = null;
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const getAudioInputDevices = async () => {
        console.log("AudioVADClient: Attempting to enumerate audio input devices.");
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                console.warn("AudioVADClient: navigator.mediaDevices.enumerateDevices() not supported.");
                setAvailableMics([{ deviceId: '', label: 'Default Microphone' }]);
                setSelectedMicId(''); // Ensure default is selected if enumeration not supported
                return;
            }
            // It's good practice to request permission before enumerating to get full labels,
            // but for simplicity in initial load, we'll enumerate directly.
            // Labels might be empty if permission hasn't been granted yet for this origin.
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioInputDevices = devices.filter(device =>  device.kind === 'audioinput');
            
            const micOptions = [
                { deviceId: '', label: 'Default Microphone' }, 
                ...audioInputDevices.map(device => ({
                    deviceId: device.deviceId,
                    label: device.label || `Microphone (${device.deviceId.substring(0, 8)}...)` 
                }))
            ];
            setAvailableMics(micOptions);

            // If no mic is selected, or if current selection is no longer valid, default to 'Default Microphone'.
            const currentSelectionStillValid = micOptions.some(mic => mic.deviceId === selectedMicId);
            if (!selectedMicId || !currentSelectionStillValid) {
                 setSelectedMicId(''); 
            }
        } catch (err) {
            console.error('AudioVADClient: Error enumerating audio devices:', err);
            setAvailableMics([{ deviceId: '', label: 'Default Microphone (Error Enumerating)' }]);
            setSelectedMicId('');
        }
    };

    // Effect to load microphone list on mount
    useEffect(() => {
        getAudioInputDevices();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); 

    const connectVadWebSocket = () => { // Renamed from connectWebSocket
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            console.log("VAD WebSocket already connected.");
            return;
        }

        console.log(`Attempting to connect to VAD WebSocket server at ${VAD_SERVER_URL}`);
        if(socketRef.current) {
            socketRef.current.close(); // Close any existing connection
        }
        
        setServerStatus('Connecting...'); // VAD server status
        socketRef.current = new WebSocket(VAD_SERVER_URL);

        socketRef.current.onopen = () => {
            console.log('VAD WebSocket connected');
            setServerStatus('Connected'); // VAD server status
            // VAD status will be updated by messages from the server
        };

        socketRef.current.onmessage = (event) => {
            // IMPORTANT ASSUMPTION: VAD server sends JSON messages
            // e.g., {"type": "VAD_SPEECH_START"} or {"type": "VAD_SPEECH_END"} or {"type": "VAD_STATUS", "message": "..."}
            try {
                const vadData = JSON.parse(event.data);
                console.log('AudioVADClient: VAD Data from server:', vadData);
                
                if (vadData.type === "VAD_SPEECH_START") {
                    setVadStatus('Speech Detected');
                    audioBufferForSttRef.current = []; // Start new buffer for STT
                    isSpeechSegmentActiveRef.current = true;
                    // setTranscription(''); // Clear previous transcription when new speech starts
                } else if (vadData.type === "VAD_SPEECH_END") {
                    setVadStatus('Speech Ended');
                    isSpeechSegmentActiveRef.current = false;
                    if (sttSocketRef.current && sttSocketRef.current.readyState === WebSocket.OPEN && audioBufferForSttRef.current.length > 0) {
                        // Concatenate all Int16Array chunks
                        let totalLength = 0;
                        audioBufferForSttRef.current.forEach(chunk => totalLength += chunk.length);
                        
                        if (totalLength > 0) {
                            const concatenatedAudio = new Int16Array(totalLength);
                            let offset = 0;
                            audioBufferForSttRef.current.forEach(chunk => {
                                concatenatedAudio.set(chunk, offset);
                                offset += chunk.length;
                            });

                            console.log(`Sending ${concatenatedAudio.length} audio samples (${(totalLength * 2 / 1024).toFixed(2)} KB) to STT server.`);
                            sttSocketRef.current.send(concatenatedAudio.buffer);
                            sttSocketRef.current.send("END_OF_STREAM"); // Signal end of this audio segment
                            setVadStatus('Segment sent for transcription...');
                        } else {
                             console.log("No audio buffered for STT for this speech segment (totalLength is 0).");
                        }
                    } else {
                        if (!sttSocketRef.current || sttSocketRef.current.readyState !== WebSocket.OPEN) {
                             console.warn("STT WebSocket not open. Cannot send audio for transcription.");
                             setVadStatus('STT server not ready for transcription.');
                        }
                        if (audioBufferForSttRef.current.length === 0) {
                            console.log("No audio buffered for STT for this speech segment (buffer empty).");
                        }
                    }
                    audioBufferForSttRef.current = []; // Clear buffer after sending or if not sent
                } else if (vadData.type === "VAD_STATUS") {
                    setVadStatus(vadData.message || 'VAD Status Update');
                } else {
                    // Fallback for other message types or non-JSON messages
                    console.warn('AudioVADClient: Received unhandled VAD message type or non-JSON:', vadData);
                    setVadStatus(typeof event.data === 'string' ? event.data : JSON.stringify(event.data));
                }
            } catch (error) {
                console.error("Error parsing VAD message or VAD message was not JSON:", event.data, error);
                setVadStatus(event.data); // Display raw data if not JSON
            }
        };

        socketRef.current.onerror = (error) => {
            console.error('WebSocket Error:', error);
            setServerStatus('Connection Error');
            setVadStatus('Server error');
            // Auto-stop recording if it was active and connection drops
            if (isRecordingRef.current) { // Use ref here
                stopRecordingLogic();
                setIsRecording(false);
            }
        };

        socketRef.current.onclose = (event) => {
            console.log('VAD WebSocket disconnected:', event.reason || 'No reason given');
            setServerStatus(`Disconnected: ${event.reason || 'Closed'}`); // VAD server status
            setVadStatus('Disconnected from server');
            isSpeechSegmentActiveRef.current = false; // Reset speech segment flag
            if (isRecordingRef.current) { // Use ref here
                stopRecordingLogic();
                setIsRecording(false);
            }
        };
    };

    // NEW: connectSttWebSocket function
    const connectSttWebSocket = () => {
        if (sttSocketRef.current && sttSocketRef.current.readyState === WebSocket.OPEN) {
            console.log("STT WebSocket already connected.");
            return;
        }
        console.log(`Attempting to connect to STT WebSocket server at ${STT_SERVER_URL}`);
        if(sttSocketRef.current) {
            sttSocketRef.current.close();
        }
        setSttServerStatus('Connecting...');
        sttSocketRef.current = new WebSocket(STT_SERVER_URL);

        sttSocketRef.current.onopen = () => {
            console.log('STT WebSocket connected');
            setSttServerStatus('Connected');
        };

        sttSocketRef.current.onmessage = (event) => {
            // Assume STT server sends JSON: {"transcription": "...", "error": "..."}
            try {
                const sttResult = JSON.parse(event.data);
                console.log('AudioVADClient: STT Result from server:', sttResult);
                if (sttResult.error) {
                    console.error('STT Server Error:', sttResult.error);
                    setTranscription(prev => `${prev}\n[STT Error: ${sttResult.error}]`.trim());
                } else {
                    // Append new transcription to existing ones for the session, or replace if desired
                    setTranscription(prev => `${prev} ${sttResult.transcription}`.trim());
                }
                setVadStatus('Transcription received.'); // Update main status to reflect STT activity
            } catch (error) {
                console.error("Error parsing STT message or STT message was not JSON:", event.data, error);
                setTranscription(prev => `${prev}\n[Raw STT Data: ${event.data}]`.trim());
            }
        };

        sttSocketRef.current.onerror = (error) => {
            console.error('STT WebSocket Error:', error);
            setSttServerStatus('Connection Error');
            setTranscription(prev => `${prev}\n[STT Server connection error.]`.trim());
        };

        sttSocketRef.current.onclose = (event) => {
            console.log('STT WebSocket disconnected:', event.reason || 'No reason given');
            setSttServerStatus(`Disconnected: ${event.reason || 'Closed'}`);
        };
    };

    const startRecording = async () => {
        console.log("AudioVADClient: startRecording called");
        if (isRecordingRef.current) return; // Use ref

        if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
            console.warn("AudioVADClient: VAD WebSocket not open. Cannot start recording.");
            setVadStatus('VAD Server not connected. Try connecting first.');
            return;
        }
        // Optionally check STT connection too, but VAD can run without STT
        if (!sttSocketRef.current || sttSocketRef.current.readyState !== WebSocket.OPEN) {
            console.warn("AudioVADClient: STT WebSocket not open. Transcription will not be available until STT server is connected.");
            // setSttServerStatus('Not Connected. Transcription disabled.'); // Or try to connect
        }

        try {
            setVadStatus('Initializing microphone...');
            // Use selectedMicId for constraints. If selectedMicId is '', it uses the default device.
            const constraints = {
                audio: selectedMicId ? { deviceId: { exact: selectedMicId } } : true,
                video: false
            };
            console.log("AudioVADClient: Using getUserMedia constraints:", constraints);
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            mediaStreamRef.current = stream;

            // After successfully getting the stream, labels for devices might be available if they weren't before.
            // Optionally, re-fetch devices here to update labels, though it might cause a quick UI flicker.
            // For simplicity, we'll skip re-fetching here for now. User can re-select if labels update.
            // await getAudioInputDevices(); // Example: if you want to refresh labels post-permission

            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: TARGET_SAMPLE_RATE
            });
            
            await audioContextRef.current.resume(); // Ensure context is running

            const source = audioContextRef.current.createMediaStreamSource(stream);
            
            scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(
                SCRIPT_PROCESSOR_BUFFER_SIZE, // bufferSize
                1,                            // inputChannels
                1                             // outputChannels
            );

            scriptProcessorRef.current.onaudioprocess = (audioProcessingEvent) => {
                if (!isRecordingRef.current) return; // Check master recording flag

                const inputBuffer = audioProcessingEvent.inputBuffer;
                const pcmDataFloat32 = inputBuffer.getChannelData(0); // Float32, range -1.0 to 1.0

                // Convert Float32 to Int16
                const pcmDataInt16 = new Int16Array(pcmDataFloat32.length);
                for (let i = 0; i < pcmDataFloat32.length; i++) {
                    pcmDataInt16[i] = Math.max(-1, Math.min(1, pcmDataFloat32[i])) * 32767;
                }
                
                // Send to VAD server
                if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
                    // console.log("AudioVADClient: Sending audio data chunk to VAD server..."); // Can be very noisy
                    socketRef.current.send(pcmDataInt16.buffer);
                } else {
                    // This log can be noisy if VAD disconnects during recording
                    // console.log(`AudioVADClient: VAD socket not ready. Not sending. isRecordingRef: ${isRecordingRef.current}, vadSocketReadyState: ${socketRef.current ? socketRef.current.readyState : 'null'}`);
                }

                // Buffer for STT if speech segment is active (based on VAD server signals)
                if (isSpeechSegmentActiveRef.current) {
                    // Create a copy of pcmDataInt16 for the buffer, as the underlying ArrayBuffer might be reused or sent.
                    audioBufferForSttRef.current.push(pcmDataInt16.slice());
                }
            };

            source.connect(scriptProcessorRef.current);
            scriptProcessorRef.current.connect(audioContextRef.current.destination); // Necessary for ScriptProcessorNode to fire

            setIsRecording(true); // This will trigger isRecordingRef.current update via useEffect
            setVadStatus('Recording...');
            setTranscription(''); // Clear previous transcriptions on new recording start
            console.log("AudioVADClient: Recording started successfully.");
        } catch (err) {
            console.error('Error starting recording:', err);
            setVadStatus(`Mic Error: ${err.message}`);
            setIsRecording(false); // Ensure state is updated
            stopRecordingLogic(); // Clean up any partial setup
        }
    };

    const stopRecordingLogic = () => {
        isSpeechSegmentActiveRef.current = false; // Ensure this is reset
        // If there's any buffered audio for STT when recording stops abruptly,
        // it's currently discarded if VAD_SPEECH_END wasn't received and processed.
        // Depending on requirements, one might choose to send it here.
        audioBufferForSttRef.current = []; 

        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(track => track.stop());
            mediaStreamRef.current = null;
        }
        if (scriptProcessorRef.current) {
            scriptProcessorRef.current.disconnect();
            scriptProcessorRef.current.onaudioprocess = null; // Important to remove the callback
            scriptProcessorRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close().catch(e => console.error("Error closing AudioContext:", e));
            audioContextRef.current = null;
        }
        console.log("AudioVADClient: stopRecordingLogic executed.");
    };

    const stopRecording = () => {
        if (!isRecording) return; // Check state, button disabled based on this

        console.log("AudioVADClient: stopRecording called by user");
        
        // Explicitly set isRecording to false. This will update isRecordingRef via its useEffect.
        setIsRecording(false); 
        stopRecordingLogic(); // Call the core logic to stop audio processing and cleanup resources

        // Note: If user stops mid-speech, the current VAD server logic (if not updated)
        // might not send a VAD_SPEECH_END. The audio buffered for STT up to this point
        // would be cleared by stopRecordingLogic without being sent.
        // For a more robust solution, one might send any pending audio here,
        // or the VAD server could detect abrupt client disconnects during speech.

        setVadStatus('Recording stopped. Ready.');
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
            <h2>Real-time VAD + STT Client (React + Silero VAD + Whisper)</h2>
            
            <div>
                <button onClick={connectVadWebSocket} disabled={serverStatus === 'Connected' || serverStatus === 'Connecting...'}>
                    Connect to VAD Server
                </button>
                <span style={{ marginLeft: '10px' }}>VAD Server: <strong>{serverStatus}</strong></span>
            </div>

            {/* NEW: STT Server Connection UI */}
            <div style={{ marginTop: '10px' }}>
                <button onClick={connectSttWebSocket} disabled={sttServerStatus === 'Connected' || sttServerStatus === 'Connecting...'}>
                    Connect to STT Server
                </button>
                <span style={{ marginLeft: '10px' }}>STT Server: <strong>{sttServerStatus}</strong></span>
            </div>

            {/* NEW: Microphone Selection Dropdown */}
            <div style={{ marginTop: '10px' }}>
                <label htmlFor="mic-select" style={{ marginRight: '5px' }}>Microphone: </label>
                <select 
                    id="mic-select"
                    value={selectedMicId} 
                    onChange={(e) => setSelectedMicId(e.target.value)}
                    disabled={isRecording || availableMics.length <= 1 && availableMics[0]?.label.includes("Error")} // Disable if recording or only error/default shown
                    style={{ minWidth: '200px', padding: '5px' }}
                >
                    {availableMics.map(mic => (
                        <option key={mic.deviceId || 'default-mic-option'} value={mic.deviceId}>
                            {mic.label}
                        </option>
                    ))}
                </select>
            </div>

            <div style={{ marginTop: '20px' }}>
                <button 
                    onClick={startRecording} 
                    disabled={isRecording || serverStatus !== 'Connected' /* Optional: || sttServerStatus !== 'Connected' */}
                >
                    Start Recording
                </button>
                <button onClick={stopRecording} disabled={!isRecording} style={{ marginLeft: '10px' }}>
                    Stop Recording
                </button>
            </div>
            <p style={{ marginTop: '20px', fontSize: '1.2em' }}>
                VAD Status: <strong style={{ color: vadStatus.toLowerCase().includes('speech') ? 'green' : (vadStatus.toLowerCase().includes('error') ? 'red' : 'black') }}>{vadStatus}</strong>
            </p>
            {/* NEW: Transcription Display */}
            <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px', minHeight: '100px', background: '#414141', whiteSpace: 'pre-wrap' }}>
                <strong>Transcription:</strong>
                <p>{transcription || "..."}</p>
            </div>
            <p>
                <small>Ensure Python VAD server (port {VAD_SERVER_URL.split(':')[2]}) is running and sends JSON messages (e.g., `{"{type: 'VAD_SPEECH_START'}"}`).</small><br/>
                <small>Ensure Python STT server is running on {STT_SERVER_URL}</small>
            </p>
        </div>
    );
}

export default AudioVADClient;
