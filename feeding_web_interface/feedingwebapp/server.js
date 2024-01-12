/**
 * Adapted from https://github.com/coding-with-chaim/webrtc-one-to-many/tree/master
 *
 * This is a barebones WebRTC server that enables one-to-many video streaming.
 * Specifically, for every topic, it enables one publisher and many subscribers.
 * Each subscriber sees the video stream from the publisher.
 */

const express = require('express')
const app = express()
const bodyParser = require('body-parser')
var cors = require('cors')
const webrtc = require('wrtc')

let senderStream = {}

app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true }))
app.use(cors())

app.post('/subscribe', async ({ body }, res) => {
  console.log('got subscriber on topic: ' + body.topic)

  // Configure the peer connection
  const peer = new webrtc.RTCPeerConnection({
    iceServers: [
      {
        urls: 'stun:stun.stunprotocol.org'
      }
    ]
  })
  const desc = new webrtc.RTCSessionDescription(body.sdp)
  await peer.setRemoteDescription(desc)

  // Add the publisher's video stream to the subscriber's peer connection
  const topic = body.topic
  if (topic in senderStream) {
    senderStream[topic].getTracks().forEach((track) => peer.addTrack(track, senderStream[topic]))
  }

  // Create an answer to the publisher's offer
  const answer = await peer.createAnswer()
  await peer.setLocalDescription(answer)
  const payload = {
    sdp: peer.localDescription
  }

  // Send the answer to the publisher
  res.json(payload)
})

app.post('/publish', async ({ body }, res) => {
  console.log('got publisher on topic: ' + body.topic)

  // Configure the peer connection
  const peer = new webrtc.RTCPeerConnection({
    iceServers: [
      {
        urls: 'stun:stun.stunprotocol.org'
      }
    ]
  })

  // Send the publisher's video stream to all subscribers on that topic
  const topic = body.topic
  peer.ontrack = (e) => handleTrackEvent(e, topic)

  // Create an answer to the publisher's offer
  const desc = new webrtc.RTCSessionDescription(body.sdp)
  await peer.setRemoteDescription(desc)
  const answer = await peer.createAnswer()
  await peer.setLocalDescription(answer)
  const payload = {
    sdp: peer.localDescription
  }

  // Send the answer to the publisher
  res.json(payload)
})

function handleTrackEvent(e, topic) {
  senderStream[topic] = e.streams[0]
}

app.listen(process.env.REACT_APP_SIGNALLING_SERVER_PORT, () => console.log('Server started'))
