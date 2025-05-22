import React, { useEffect, useRef, useState } from 'react';

// Configuration - Make sure this matches your server's HTTP signaling endpoint
const SIGNALING_SERVER_URL = 'http://localhost:8767/offer_stt';

function AudioVADClient() {
  const pc = useRef(null); // RTCPeerConnection instance
  const dataChannel = useRef(null); // DataChannel for receiving VAD status
  const localStream = useRef(null); // Local media stream (microphone)

  const [connectionState, setConnectionState] = useState('disconnected');
  const [vadStatus, setVadStatus] = useState(false);
  const [error, setError] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);
  
  const operationInProgress = useRef(false); // For robust re-entrancy guard
  // Ref to manage effect execution in StrictMode for development
  const effectRan = useRef(false);

  const startWebRTC = async () => {
    // Guard against starting if already connecting or if a connection exists and isn't closed.
    if (operationInProgress.current) {
      console.log('startWebRTC: Aborting, operation already in progress.');
      return;
    }
    operationInProgress.current = true;

    // If there's an existing connection, clean it up first
    if (pc.current) {
      await stopWebRTC(true); // Indicate this stop is part of a start operation
      // Wait a brief moment for cleanup
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    setIsConnecting(true);
    setError(null);
    setVadStatus(false);
    setTranscription('');
    setConnectionState('connecting');
    console.log('startWebRTC: Initiating new connection attempt.', { isConnectingState: isConnecting });

    try {
      // 1. Get user media (microphone)
      localStream.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('Microphone access granted.');

      // 2. Create RTCPeerConnection with ICE servers
      pc.current = new RTCPeerConnection({
        iceServers: [
          // Local candidates first
          {
            urls: [
              'stun:stun.l.google.com:19302',
              'stun:stun1.l.google.com:19302',
              'stun:stun2.l.google.com:19302',
              'stun:stun3.l.google.com:19302',
              'stun:stun.stunprotocol.org:3478'
            ]
          },
          // UDP TURN servers
          {
            urls: [
              'turn:openrelay.metered.ca:80?transport=udp',
              'turn:relay.metered.ca:80?transport=udp'
            ],
            username: 'openrelayproject',
            credential: 'openrelayproject'
          },
          // TCP TURN servers (fallback)
          {
            urls: [
              'turn:openrelay.metered.ca:443?transport=tcp',
              'turn:relay.metered.ca:443?transport=tcp'
            ],
            username: 'openrelayproject',
            credential: 'openrelayproject'
          }
        ],
        iceTransportPolicy: 'all',
        iceCandidatePoolSize: 2, // Reduced to focus on quality candidates
        bundlePolicy: 'max-bundle',
        rtcpMuxPolicy: 'require'
      });
      console.log('RTCPeerConnection created with ICE servers.');

      // Connection timeout handler
      let connectionTimeoutId = null;
      const startConnectionTimeout = (timeoutMs = 15000) => {
        // Clear any existing timeout
        if (connectionTimeoutId) {
          clearTimeout(connectionTimeoutId);
        }
        
        connectionTimeoutId = setTimeout(() => {
          if (!pc.current) return; // Connection already cleaned up
          
          const state = {
            connectionState: pc.current.connectionState,
            iceConnectionState: pc.current.iceConnectionState,
            iceGatheringState: pc.current.iceGatheringState,
            signalingState: pc.current.signalingState
          };
          
          // More nuanced stall detection
          const isStalled = (
            // In 'new' state too long after gathering
            (state.iceConnectionState === 'new' && 
             state.iceGatheringState === 'complete' &&
             state.connectionState === 'new') ||
            // Or checking with no progress
            (state.iceConnectionState === 'checking' && 
             state.iceGatheringState === 'complete' && 
             state.connectionState !== 'connected') ||
            // Or failed state
            state.connectionState === 'failed'
          );

          if (isStalled) {
            console.log('Connection appears stalled. States:', {
              ...state,
              canTrickle: pc.current.canTrickleIceCandidates
            });
            setError('Connection stalled - attempting to recover...');
            
            // Try to restart ICE
            if (pc.current && pc.current.restartIce) {
              console.log('Attempting ICE restart...');
              pc.current.restartIce();
              // Give it some time with a shorter timeout
              startConnectionTimeout(10000);
              return;
            }
            
            // If ICE restart not possible, try TURN
            if (pc.current && pc.current.iceConnectionState !== 'connected') {
              console.log('Switching to TURN only...');
              pc.current.iceTransportPolicy = 'relay';
              startConnectionTimeout(10000); // shorter timeout for TURN attempt
              return;
            }
            
            // If all recovery attempts fail, stop
            console.log('Recovery attempts failed, stopping connection');
            stopWebRTC(); // This is an external stop, isPartOfStartOperation defaults to false
          }
        }, timeoutMs);
      };

      // Start the initial timeout
      startConnectionTimeout();

      // 3. Add microphone track to the connection
      localStream.current.getTracks().forEach(track => {
        pc.current.addTrack(track, localStream.current);
        console.log('Microphone track added.');
      });

      // 4. Set up event handlers
      pc.current.oniceconnectionstatechange = () => {
        if (!pc.current) return;
        console.log('ICE connection state changed:', pc.current.iceConnectionState);
        
        // Restart timeout on state changes
        if (pc.current.iceConnectionState === 'checking' || 
            pc.current.iceConnectionState === 'new') {
          startConnectionTimeout();
        }
      };

      pc.current.onicegatheringstatechange = () => {
        if (!pc.current) return;
        const state = pc.current.iceGatheringState;
        console.log('ICE gathering state changed:', state);
        
        if (state === 'complete') {
          // Give a short window for the connection to establish after gathering
          startConnectionTimeout();
        }
      };

      pc.current.onicecandidate = event => {
        if (event.candidate) {
          console.log('New ICE candidate:', {
            type: event.candidate.type,
            protocol: event.candidate.protocol,
            address: event.candidate.address,
            port: event.candidate.port,
            component: event.candidate.component,
          });
        } else {
          console.log('ICE gathering complete - null candidate received');
        }
      };

      pc.current.onconnectionstatechange = () => {
        // If pc.current has been nulled by stopWebRTC (e.g., due to another cleanup), bail.
        if (!pc.current) {
            console.log('onconnectionstatechange: pc.current is null, bailing.');
            return;
        }
        console.log('Connection state changes:', {
          connectionState: pc.current.connectionState,
          iceConnectionState: pc.current.iceConnectionState,
          iceGatheringState: pc.current.iceGatheringState,
          signalingState: pc.current.signalingState
        });
        setConnectionState(pc.current.connectionState);
        
        const currentPcState = pc.current.connectionState;
        if (currentPcState === 'failed' || currentPcState === 'disconnected' || currentPcState === 'closed') {
            console.log(`onconnectionstatechange: PC state is ${currentPcState}, calling stopWebRTC.`);
            // This is an external stop, isPartOfStartOperation defaults to false
            (async () => { 
              await stopWebRTC(); // Clean up on failure, disconnection or explicit close
            })().catch(e => console.error('Error in connection state cleanup:', e));
        }
      };

      pc.current.ondatachannel = event => {
        // This handler is for data channels INITIATED BY THE REMOTE PEER (the server)
        console.log('Data channel received from server:', event.channel.label);
        const channel = event.channel;
        
        // Common handlers for both channel types
        channel.onopen = () => {
          console.log(`Data channel '${channel.label}' opened.`);
        };

        channel.onclose = () => {
          console.log(`Data channel '${channel.label}' closed.`);
          if (channel.label === 'vad_status_feed') {
            setVadStatus(false); // Reset VAD status on VAD channel close
            if (dataChannel.current === channel) {
              dataChannel.current = null;
            }
          }
        };

        channel.onerror = err => {
          console.error(`Data channel '${channel.label}' error:`, err);
          setError(`Data channel error (${channel.label})`);
        };

        if (channel.label === 'vad_status_feed') {
          dataChannel.current = channel; // Store VAD channel reference

          event.channel.onmessage = msgEvent => {
            try {
              const data = JSON.parse(msgEvent.data);
              if (data.vad_status !== undefined && typeof data.vad_status === 'boolean') {
                // console.log('Received VAD status:', data.vad_status); // Keep this if you want separate VAD status logs
                setError(null); // Clear previous general errors if VAD status is coming through
                setVadStatus(data.vad_status);
              } else if (data.type === "stt_transcription" && data.data) {
                // Log all transcription data for debugging
                console.log('Received transcription data:', data.data);

                // Handle transcription text if present
                if (data.data.transcription) {
                  console.log('Received transcription text:', data.data.transcription);
                  // Append new transcription with a space
                  setTranscription(prev => prev + data.data.transcription + " ");
                  // Clear any previous STT errors when we get valid transcription
                  setError(prev => prev?.startsWith("STT:") ? null : prev);
                }

                // Handle errors if present
                if (data.data.error) {
                  console.warn('STT Error received:', data.data.error);
                  // Set error only if it's meaningful
                  if (data.data.error !== "Empty audio buffer for transcription") {
                    setError(`STT: ${data.data.error}`);
                  }
                }
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

      // 5. Create and set local description, then handle signaling
      const offer = await pc.current.createOffer();
      console.log('Offer created, setting local description...');
      
      // Set local description and ensure it completes
      await pc.current.setLocalDescription(offer);
      console.log('Local description set, state:', pc.current.signalingState);

      // Verify the connection still exists and is in the correct state
      if (!pc.current) {
        throw new Error('Connection was cleared after setting local description');
      }

      if (pc.current.signalingState !== 'have-local-offer') {
        console.warn('Unexpected signaling state before sending offer:', pc.current.signalingState);
        throw new Error('Connection in invalid state for sending offer');
      }

      // 6. Send offer to server
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

      // 7. Set remote description
      // Double check the connection still exists and is in the correct state
      if (!pc.current) {
        throw new Error('Connection was cleared while waiting for answer');
      }

      if (pc.current.signalingState !== 'have-local-offer') {
        console.warn('Unexpected signaling state before setting remote description:', 
          pc.current.signalingState);
        throw new Error('Connection in invalid state for setting remote description');
      }

      try {
        // Create and set the remote description
        const remoteDesc = new RTCSessionDescription(answer);
        await pc.current.setRemoteDescription(remoteDesc);
        console.log('Remote description set successfully, state:', pc.current.signalingState);
      } catch (e) {
        console.error('Failed to set remote description:', e, 
          'Current state:', pc.current.signalingState);
        throw e;
      }

      // 8. Now wait for ICE gathering and connection establishment
      await Promise.all([
        // Wait for ICE connection
        new Promise((resolve, reject) => {
          let overallTimeoutId;
          let retryCount = 0;
          const maxRetries = 2;
          const originalIceConnectionStateChange = pc.current?.oniceconnectionstatechange;

          const cleanup = () => {
            clearTimeout(overallTimeoutId);
            if (pc.current) { // Restore if pc.current exists
              pc.current.oniceconnectionstatechange = originalIceConnectionStateChange; // Restore original handler
            }
          };

          const handleIceStateChange = () => {
            if (!pc.current) {
              cleanup();
              reject(new Error('Connection was cleared during ICE negotiation.'));
              return;
            }

            const currentIceState = pc.current.iceConnectionState;
            console.log('Promise: ICE connection state changed:', currentIceState, { retryCount });

            if (currentIceState === 'connected' || currentIceState === 'completed') {
              cleanup();
              resolve();
            } else if (currentIceState === 'failed') {
              if (retryCount < maxRetries) {
                retryCount++;
                console.log(`Promise: ICE connection failed. Retry ${retryCount}/${maxRetries}.`);
                if (pc.current && pc.current.restartIce) {
                  console.log('Promise: Attempting ICE restart...');
                  pc.current.restartIce();
                  // The state change from restartIce (e.g., back to 'checking' or 'new')
                  // will trigger this handler again.
                } else {
                  console.warn("Promise: restartIce not available for retry. Cannot proceed with this retry mechanism.");
                  // If restartIce isn't available, this specific retry path ends.
                  // The overall timeout will eventually fire if no other state change leads to resolution.
                }
              } else {
                console.error('Promise: ICE connection failed after max retries.');
                cleanup();
                reject(new Error('ICE connection failed after retries.'));
              }
            } else if (currentIceState === 'disconnected' || currentIceState === 'closed') {
                console.log(`Promise: ICE state is ${currentIceState}. Connection lost/closed.`);
                cleanup();
                reject(new Error(`ICE connection ${currentIceState}.`));
            }
            // Other states ('new', 'checking') are intermediate. Wait for them to change or for overall timeout.
          };

          if (!pc.current) {
            reject(new Error('PeerConnection not available at start of ICE promise.'));
            return;
          }

          // Handle ICE connection state changes
          pc.current.oniceconnectionstatechange = handleIceStateChange;

          // Overall timeout for this ICE connection phase
          overallTimeoutId = setTimeout(() => {
            // Check pc.current again as it might have been cleaned up by another process
            if (pc.current && pc.current.iceConnectionState !== 'connected' && pc.current.iceConnectionState !== 'completed') {
              console.error('Promise: ICE connection attempt timed out globally.', {
                state: pc.current.iceConnectionState,
                gatheringState: pc.current.iceGatheringState
              });
              cleanup();
              reject(new Error(`ICE connection timed out (state: ${pc.current.iceConnectionState}, gathering: ${pc.current.iceGatheringState})`));
            }
            // If it connected/completed just before timeout, resolve() would have cleared this timeout.
          }, 35000); // Increased timeout (e.g., 35 seconds) for the entire ICE connection phase, including retries.

          // Initial check
          handleIceStateChange();
        }),

        // Monitor ICE gathering (but don't block on it)
        new Promise((resolve) => {
          let candidateCount = 0;
          const handleCandidate = event => {
            if (event.candidate) {
              candidateCount++;
              console.log('ICE candidate found:', {
                type: event.candidate.type,
                protocol: event.candidate.protocol,
                address: event.candidate.address,
                count: candidateCount
              });
            } else {
              console.log('ICE gathering complete, total candidates:', candidateCount);
              resolve();
            }
          };

          if (pc.current.iceGatheringState === 'complete') {
            resolve();
          } else {
            pc.current.onicecandidate = handleCandidate;
          }
        })
      ]);

      console.log('Connection established successfully');

    } catch (e) {
      console.error('WebRTC connection failed:', e);
      setError(`Connection failed: ${e.message}`);
      setConnectionState('failed');
      // This stop is due to an error in the current startWebRTC flow.
      // The finally block will handle operationInProgress.current.
      await stopWebRTC(true); 
    } finally {
        setIsConnecting(false);
        // This ensures the lock is always released when startWebRTC finishes,
        // regardless of success or failure, or how stopWebRTC was called internally.
        operationInProgress.current = false;
        console.log('startWebRTC: Connection attempt finished (finally block).');
    }
  };

  const stopWebRTC = async (isPartOfStartOperation = false) => {
    console.log(`stopWebRTC: Attempting to stop WebRTC connection... (isPartOfStartOperation: ${isPartOfStartOperation})`);
    setIsConnecting(false); 

    if (!isPartOfStartOperation) {
      // Only reset the global operation lock if this stop is "external"
      // and not part of an ongoing startWebRTC that will manage the lock itself.
      console.log('stopWebRTC: Resetting operationInProgress flag (not part of start).');
      operationInProgress.current = false;
    } else {
      console.log('stopWebRTC: Not resetting operationInProgress flag (part of start).');
    }

    // Clean up data channel first
    if (dataChannel.current) {
      console.log('Cleaning up data channel...');
      try {
        // Remove event handlers before closing
        dataChannel.current.onopen = null;
        dataChannel.current.onmessage = null;
        dataChannel.current.onclose = null;
        dataChannel.current.onerror = null;
        
        if (dataChannel.current.readyState !== 'closed') {
          dataChannel.current.close();
        }
      } catch (e) {
        console.warn('Error cleaning up data channel:', e);
      }
      dataChannel.current = null;
    }

    // Clean up peer connection
    const pcToClose = pc.current; // Capture current PC reference
    if (pcToClose) {
      console.log('Cleaning up peer connection...', {
        connectionState: pcToClose.connectionState,
        signalingState: pcToClose.signalingState,
        iceConnectionState: pcToClose.iceConnectionState,
        iceGatheringState: pcToClose.iceGatheringState
      });

      try {
        // Remove all event handlers
        pcToClose.onicecandidate = null;
        pcToClose.oniceconnectionstatechange = null;
        pcToClose.onicegatheringstatechange = null;
        pcToClose.onconnectionstatechange = null;
        pcToClose.onsignalingstatechange = null;
        pcToClose.ondatachannel = null;
        pcToClose.ontrack = null;
        
        // Remove all tracks
        pcToClose.getSenders().forEach(sender => {
          try {
            pcToClose.removeTrack(sender);
          } catch (e) {
            console.warn('Error removing track:', e);
          }
        });

        // Close the connection if not already closed
        if (pcToClose.signalingState !== 'closed') {
          pcToClose.close();
          console.log('Peer connection closed.');
        }
      } catch (e) {
        console.warn('Error during peer connection cleanup:', e);
      }

      // Clear the reference if it still points to the one we just cleaned up
      if (pc.current === pcToClose) {
        pc.current = null;
      }
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
  // eslint-disable-next-line react-hooks/exhaustive-deps -- We want this effect to run only on mount
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
      // Create an async function and immediately execute it
      (async () => {
        await stopWebRTC();
      })().catch(e => console.error('Error in useEffect cleanup:', e));
      // Do NOT reset effectRan.current to false here for StrictMode's mount/unmount/mount cycle.
      // It should remain true to indicate that for this component instance's "true" mount,
      // the setup effect has already run. It will naturally be false if the component truly unmounts and a new instance is later mounted.
    };
  }, []); // Empty dependency array: runs on mount and unmount (or simulated unmount/remount in StrictMode)

  return (
    <div>
      <h2>WebRTC VAD Client</h2>
      <p>Connection State: <strong>{connectionState}</strong></p>
      <p>VAD Status: <strong style={{ color: vadStatus ? 'green' : 'red' }}>{vadStatus ? 'Speech Detected' : 'Silence'}</strong></p>
      <div>
        <label htmlFor="transcriptionArea" style={{ display: 'block', marginBottom: '5px' }}>Transcription:</label>
        <textarea
          id="transcriptionArea"
          value={transcription || "..."}
          readOnly
          style={{ width: '90%', minHeight: '100px', padding: '5px', border: '1px solid #ccc', borderRadius: '4px', backgroundColor: '#f9f9f9' }}
        />
      </div>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      <div>
        {connectionState === 'disconnected' || connectionState === 'failed' ? (
          <button onClick={() => startWebRTC().catch(e => console.error('Button onClick: Error starting WebRTC:', e))} disabled={isConnecting}>
            {isConnecting ? 'Connecting...' : 'Start VAD'}
          </button>
        ) : (
          <button onClick={() => stopWebRTC().catch(e => console.error('Button onClick: Error stopping WebRTC:', e))} disabled={connectionState === 'closing' || isConnecting}>
            Stop VAD
          </button>
        )}
      </div>
    </div>
  );
}

export default AudioVADClient;