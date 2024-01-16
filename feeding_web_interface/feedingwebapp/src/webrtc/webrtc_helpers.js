// Code adapted from https://github.com/hcrlab/stretch_teleop_interface/blob/master/src/shared/webrtcconnections.tsx

import axios from 'axios'

/**
 * Creates a peer connection to the given url and topic. This is intended to work with
 * the server in server.js.
 *
 * @param {string} url The url of the server to connect to.
 * @param {string} topic The topic to subscribe to.
 * @param {function} onTrackAdded The callback function to call when a track is added.
 * @param {function} onConnectionEnd The callback function to call when the connection ends.
 * @returns {object} The RTCPeerConnection.
 */
export function createPeerConnection(url, topic, onTrackAdded, onConnectionEnd) {
  try {
    const peerConnection = new RTCPeerConnection({
      iceServers: [
        {
          urls: 'stun:stun1.l.google.com:19302'
        }
      ]
    })

    peerConnection.ontrack = onTrackAdded

    peerConnection.onnegotiationneeded = async () => {
      try {
        const offer = await peerConnection.createOffer()
        await peerConnection.setLocalDescription(offer)
        const ip = await getIPAddress()
        const payload = {
          sdp: peerConnection.localDescription,
          topic: topic,
          ip: ip
        }
        console.log('sending payload', payload)
        const { data } = await axios.post(url, payload)
        const desc = new RTCSessionDescription(data.sdp)
        peerConnection.setRemoteDescription(desc).catch((e) => console.log(e))
      } catch (e) {
        console.log(e)
      }
    }

    peerConnection.oniceconnectionstatechange = () => {
      if (!peerConnection) throw new Error('peerConnection is undefined')
      if (peerConnection.iceConnectionState === 'failed') {
        peerConnection.restartIce()
      }
    }

    peerConnection.onconnectionstatechange = () => {
      if (!peerConnection) throw new Error('peerConnection is undefined')
      if (peerConnection.connectionState === 'failed' || peerConnection.connectionState === 'disconnected') {
        console.error(peerConnection.connectionState, 'Resetting the PeerConnection')
        if (onConnectionEnd) onConnectionEnd()
        // peerConnection = createPeerConnection(url, topic, onTrackAdded, onConnectionEnd)
      }
      console.log('peerConnection.onconnectionstatechange', peerConnection.connectionState)
    }

    peerConnection.onicecandidateerror = (event) => {
      console.error('ICE candidate gathering error:', event.errorCode)
    }

    console.log('Created RTCPeerConnection')
    return peerConnection
  } catch (err) {
    console.error('Failed to create PeerConnection, exception: ' + err.message)
    return
  }
}

/**
 * Closes the given peer connection.
 * @param {object} peerConnection The RTCPeerConnection to close.
 */
export function closePeerConnection(peerConnection) {
  if (!peerConnection) return
  console.log('Closing RTCPeerConnection', peerConnection)
  if (peerConnection.connectionState !== 'closed') {
    const senders = peerConnection.getSenders()
    senders.forEach((sender) => peerConnection.removeTrack(sender))
    peerConnection.close()
    console.log('Closed RTCPeerConnection')
  }
}

export async function getIPAddress() {
  const res = await axios.get('https://api.ipify.org/?format=json')
  console.log(res.data)
  return res.data.ip
}
