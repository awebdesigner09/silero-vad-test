import React, { useEffect, useRef, useState } from 'react';

// Configuration - Make sure this matches your server's HTTP signaling endpoint
const SIGNALING_SERVER_URL = 'http://localhost:8766/offer'; // Server runs on PORT + 1 (8765 + 1)

function AudioVADClient() {
  const pc = useRef(null); // RTCPeerConnection instance
  const dataChannel = useRef(null); // DataChannel for receiving VAD status
  const localStream = useRef(null); // Local media stream (microphone)

  const [connectionState, setConnectionState] = useState('disconnected');
  const [vadStatus, setVadStatus] = useState(false);
  const [error, setError] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);
  
  // Ref to manage effect execution in StrictMode for development
  const effectRan = useRef(false);

  const startWebRTC = async () => {
    // Guard against starting if already connecting or if a connection exists and isn't closed.
    if (isConnecting) {
      console.log('startWebRTC: Aborting, connection attempt already in progress (isConnecting).');
      return;
    }    if (pc.current && pc.current.connectionState !== 'closed') {
      console.log('Connection already exists or is in progress.');
      return;
    }

    setIsConnecting(true);
    setError(null);
    setVadStatus(false);
    setTranscription('');
    setConnectionState('connecting');
    console.log('startWebRTC: Initiating new connection attempt.');

    try {
      // 1. Get user media (microphone)
      localStream.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('Microphone access granted.');

      // 2. Create RTCPeerConnection
      // Ensure any old pc is cleaned up if this is a restart. stopWebRTC should handle this,
      // but explicitly creating a new one here is key.
      pc.current = new RTCPeerConnection();
      console.log('RTCPeerConnection created.');

      // 3. Add microphone track to the connection
      localStream.current.getTracks().forEach(track => {
        pc.current.addTrack(track, localStream.current);
        console.log('Microphone track added.');
      });

      // 4. Set up event handlers
      pc.current.onicecandidate = event => {
        if (event.candidate) {
          // Send ICE candidates to the server if needed (optional for simple setups)
          // console.log('ICE candidate:', event.candidate);
          // Note: For this simple example, we rely on trickle ICE being handled
          // implicitly by the SDP exchange or not strictly required for localhost.
          // In production, you'd send candidates to the server here.
        }
      };

      pc.current.onconnectionstatechange = () => {
        // If pc.current has been nulled by stopWebRTC (e.g., due to another cleanup), bail.
        if (!pc.current) {
            console.log('onconnectionstatechange: pc.current is null, bailing.');
            return;
        }
        console.log('Connection state changed:', pc.current.connectionState);
        setConnectionState(pc.current.connectionState);
        
        const currentPcState = pc.current.connectionState;
        if (currentPcState === 'failed' || currentPcState === 'disconnected' || currentPcState === 'closed') {
            console.log(`onconnectionstatechange: PC state is ${currentPcState}, calling stopWebRTC.`);
            stopWebRTC(); // Clean up on failure, disconnection or explicit close
        }
      };

      pc.current.ondatachannel = event => {
        // This handler is for data channels INITIATED BY THE REMOTE PEER (the server)
        console.log('Data channel received from server:', event.channel.label);
        if (event.channel.label === 'vad_status_feed') {
          dataChannel.current = event.channel;

          // Assign handlers to the new channel
          event.channel.onopen = () => {
            console.log('VAD status data channel opened.');
            // You could send a message to the server here if needed, e.g., a "ready" signal
            // dataChannel.current.send(JSON.stringify({ status: 'ready' }));
          };

          event.channel.onmessage = msgEvent => {
            try {
              const data = JSON.parse(msgEvent.data);
              if (data.vad_status !== undefined && typeof data.vad_status === 'boolean') {
                // console.log('Received VAD status:', data.vad_status); // Keep this if you want separate VAD status logs
                setVadStatus(data.vad_status);
              } else if (data.type === "stt_transcription" && data.data && data.data.transcription) {
                console.log('Received transcription:', data.data.transcription);
                // For continuous transcription, you might append. For now, replacing.
                // Also, consider if data.data.is_final is true to clear previous partials.
                setTranscription(prev => prev + data.data.transcription + " "); // Append new transcription parts
              }
            } catch (parseError) {
              console.error('Failed to parse VAD status message:', parseError);
            }
          };

          event.channel.onclose = () => {
            console.log('VAD status data channel closed.');
            setVadStatus(false); // Reset VAD status on channel close
            // setTranscription(''); // Optionally clear transcription
            if (dataChannel.current === event.channel) { // Clear ref if this is the active channel
              dataChannel.current = null;
            }
          };

          event.channel.onerror = err => {
            console.error('VAD status data channel error:', err);
            setError('Data channel error');
          };
        } else {
            console.warn(`Received unexpected data channel: ${event.channel.label}`);
        }
      };

      // 5. Create offer
      const offer = await pc.current.createOffer();
      await pc.current.setLocalDescription(offer);
      console.log('Offer created and set as local description.');

      // Wait for ICE gathering to complete (optional, but can help)
      // In a real app, you might use trickle ICE instead of waiting.
      // For simplicity here, we'll just proceed.
      // await new Promise(resolve => {
      //   if (pc.current.iceGatheringState === 'complete') {
      //     resolve();
      //   } else {
      //     pc.current.onicegatheringstatechange = () => {
      //       if (pc.current.iceGatheringState === 'complete') {
      //         resolve();
      //       }
      //     };
      //   }
      // });

      // 6. Send offer to server and receive answer
      console.log('Sending offer to signaling server...');
      const response = await fetch(SIGNALING_SERVER_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Signaling server error: ${response.status} ${errorText}`);
      }

      const answer = await response.json();
      console.log('Received answer from signaling server.');

      // 7. Set server's answer as remote description
       if (pc.current && (pc.current.signalingState === 'have-local-offer' || pc.current.signalingState === 'pranswer')) { // pranswer is also a valid state
            try {
                await pc.current.setRemoteDescription(new RTCSessionDescription(answer));
                console.log('Answer set as remote description. Connection should now be stable or connecting.');
            } catch (e) {
                console.error('Error setting remote description (answer):', e);
                setError(`Error setting remote answer: ${e.message}`);
                stopWebRTC(); // Clean up on error
            }
        } else {
            const currentState = pc.current ? pc.current.signalingState : 'null';
            console.error(
                `Cannot set remote description. PeerConnection is in wrong state: ${currentState} (expected have-local-offer or pranswer) or pc.current is null.`
            );
            // This indicates a problem with how startWebRTC is called or how pc.current is managed.
            // You might want to call stopWebRTC() here if the state is invalid.
            // If pc.current exists and is not already closed, and we are in a bad state, stop this attempt.
            if (pc.current && pc.current.signalingState !== 'closed') {
                stopWebRTC();
            }
            setError(`Failed to set remote answer: PC in wrong state (${currentState})`);
        }

    } catch (e) {
      console.error('WebRTC connection failed:', e);
      setError(`Connection failed: ${e.message}`);
      setConnectionState('failed');
      stopWebRTC(); // Clean up on error
    } finally {
        setIsConnecting(false);
        console.log('startWebRTC: Connection attempt finished (finally block).');
        // If connection failed, transcription might need reset if not already done
    }
  };

  const stopWebRTC = () => {
    console.log('stopWebRTC: Attempting to stop WebRTC connection...');
    setIsConnecting(false); // Ensure isConnecting is reset
        if (dataChannel.current) {
      // Remove event handlers before closing
      dataChannel.current.onopen = null;
      dataChannel.current.onmessage = null;
      dataChannel.current.onclose = null;
      dataChannel.current.onerror = null;
      dataChannel.current.close();
      dataChannel.current = null;
    }

    const pcToClose = pc.current; // Capture before potentially nullifying
    if (pcToClose) {
      console.log(`stopWebRTC: PeerConnection found. State: ${pcToClose.connectionState}, Signaling: ${pcToClose.signalingState}`);

      // Remove event handlers
      pc.current.onicecandidate = null;
      pc.current.onconnectionstatechange = null;
      pc.current.ondatachannel = null;
      // If you were using pc.current.ontrack, nullify it here too
      // pc.current.ontrack = null;

      // Only close if not already closed
      if (pcToClose.signalingState !== 'closed') {
        pcToClose.close();
        console.log('stopWebRTC: pc.close() called.');
      }
      // Nullify the ref if it's still pointing to the PC we just handled.
      // This helps prevent using a closed PC.
      if(pc.current === pcToClose) pc.current = null;
    }
    if (localStream.current) {
      localStream.current.getTracks().forEach(track => track.stop());
      localStream.current = null;
    }
    setConnectionState('disconnected');
    setVadStatus(false);
    setTranscription(''); // Clear transcription on stop
    setIsConnecting(false);
    // setError(null); // Optionally reset error on explicit stop
    console.log('stopWebRTC: Procedure finished.');
  };

  // Effect to start the connection when the component mounts
  useEffect(() => {
    // In development, React StrictMode runs effects twice to help find bugs.
    // This ref-based approach ensures startWebRTC is effectively called once on "true" mount.
    if (process.env.NODE_ENV === 'development') {
      if (effectRan.current === false) {
        console.log('useEffect: Running startWebRTC (Strict Mode - first run)');
        startWebRTC();
        effectRan.current = true; // Mark that the effect's main logic has run
      } else {
        console.log('useEffect: Skipped startWebRTC (Strict Mode - subsequent run)');
      }
    } else {
      // In production, it runs once.
      console.log('useEffect: Running startWebRTC (Production Mode)');
      startWebRTC();
    }

    // Effect cleanup function to stop the connection when the component unmounts
    return () => {
      console.log('useEffect: Cleanup function running - calling stopWebRTC.');
      stopWebRTC();
      // Do NOT reset effectRan.current to false here for StrictMode's mount/unmount/mount cycle.
      // It should remain true to indicate that for this component instance's "true" mount,
      // the setup effect has already run. It will naturally be false if the component truly unmounts and a new instance is later mounted.
    };
  }, []); // Empty dependency array: runs on mount and unmount (or simulated unmount/remount in StrictMode)

  return (
    <div>
      <h2>WebRTC VAD Client</h2>
      <p>Connection State: <strong>{connectionState}</strong></p>
      <p>VAD Status: <strong>{vadStatus ? 'Speech Detected' : 'Silence'}</strong></p>
      <p>Transcription: <strong>{transcription || "..."}</strong></p>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      <div>
        {connectionState === 'disconnected' || connectionState === 'failed' ? (
          <button onClick={startWebRTC} disabled={isConnecting}>
            {isConnecting ? 'Connecting...' : 'Start VAD'}
          </button>
        ) : (
          <button onClick={stopWebRTC} disabled={connectionState === 'closing' || isConnecting}>
            Stop VAD
          </button>
        )}
      </div>
    </div>
  );
}

export default AudioVADClient;