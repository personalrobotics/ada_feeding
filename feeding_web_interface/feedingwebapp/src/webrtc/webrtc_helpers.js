/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// Code adapted from https://github.com/hcrlab/stretch_teleop_interface/blob/master/src/shared/webrtcconnections.tsx

import axios from 'axios'

/**
 * Creates a connection to the WebRTC signalling server defined in `server.js`.
 *
 * @param {Object} options
 *   @param {string} options.url - The URL of the WebRTC signalling server.
 *   @param {string} options.topic - The topic to subscribe to.
 *   @param {function} options.onTrackAdded - The callback to run when a track is added.
 *   @param {function} options.onConnectionEnd - The callback to run when the connection ends.
 *   @param {string} options.transceiverKind - The kind of transceiver to add to the peer connection.
 *   @param {Object} options.transceiverOptions - The options for the transceiver.
 *   @param {MediaStream} options.stream - The stream to add to the peer connection.
 *
 * @returns {WebRTCConnection} The WebRTC connection.
 */
export class WebRTCConnection {
  constructor({ url, topic, onTrackAdded, onConnectionEnd, transceiverKind = null, transceiverOptions = null, stream = null }) {
    this.url = url
    this.topic = topic
    this.onTrackAdded = onTrackAdded
    this.onConnectionEnd = onConnectionEnd
    this.transceiverKind = transceiverKind
    this.transceiverOptions = transceiverOptions
    this.stream = stream

    this.createPeerConnection()
  }

  createPeerConnection() {
    try {
      this.peerConnection = new RTCPeerConnection({
        iceServers: [
          {
            urls: 'stun:stun1.l.google.com:19302'
          }
        ]
      })

      this.peerConnection.ontrack = this.onTrackAdded

      this.peerConnection.onnegotiationneeded = async () => {
        try {
          console.log('onnegotiationneeded')
          const offer = await this.peerConnection.createOffer()
          await this.peerConnection.setLocalDescription(offer)
          // const ip = await this.getIPAddress()
          const payload = {
            sdp: this.peerConnection.localDescription,
            topic: this.topic
            // ip: ip
          }
          console.log('sending payload', payload)
          const { data } = await axios.post(this.url, payload)
          const desc = new RTCSessionDescription(data.sdp)
          this.peerConnection.setRemoteDescription(desc).catch((e) => console.log(e))
        } catch (e) {
          console.log(e)
        }
      }

      this.peerConnection.oniceconnectionstatechange = () => {
        console.log('oniceconnectionstatechange')
        if (!this.peerConnection) throw new Error('peerConnection is undefined')
        if (this.peerConnection.iceConnectionState === 'failed') {
          this.peerConnection.restartIce()
        }
      }

      this.peerConnection.onconnectionstatechange = () => {
        if (!this.peerConnection) throw new Error('peerConnection is undefined')
        if (this.peerConnection.connectionState === 'failed' || this.peerConnection.connectionState === 'disconnected') {
          console.error(this.peerConnection.connectionState, 'Resetting the PeerConnection')
          if (this.onConnectionEnd) this.onConnectionEnd()
          this.createPeerConnection()
        }
        console.log('peerConnection.onconnectionstatechange', this.peerConnection.connectionState)
      }

      this.peerConnection.onicecandidateerror = (event) => {
        console.error('ICE candidate gathering error:', event.errorCode)
      }

      if (this.transceiverKind) {
        this.peerConnection.addTransceiver(this.transceiverKind, this.transceiverOptions)
      }

      if (this.stream) {
        this.stream.getTracks().forEach((track) => this.peerConnection.addTrack(track, this.stream))
      }

      console.log('Created RTCPeerConnection', this)
    } catch (err) {
      console.error('Failed to create PeerConnection, exception: ' + err.message)
    }
  }

  close() {
    if (!this.peerConnection) return
    console.log('Closing RTCPeerConnection', this.peerConnection)
    if (this.peerConnection.connectionState !== 'closed') {
      const senders = this.peerConnection.getSenders()
      senders.forEach((sender) => this.peerConnection.removeTrack(sender))
      this.peerConnection.close()
      console.log('Closed RTCPeerConnection')
    }
  }

  async getIPAddress() {
    const res = await axios.get('https://api.ipify.org/?format=json')
    console.log(res.data)
    return res.data.ip
  }
}
