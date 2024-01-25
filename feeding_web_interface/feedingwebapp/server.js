/**
 * Adapted from https://github.com/coding-with-chaim/webrtc-one-to-many/tree/master
 *
 * This is a barebones WebRTC server that enables one-to-many video streaming.
 * Specifically, for every topic, it enables one publisher and many subscribers.
 * Each subscriber sees the video stream from the publisher.
 */

const SegfaultHandler = require('segfault-handler')
SegfaultHandler.registerHandler('crash.log')
const express = require('express')
const app = express()
const bodyParser = require('body-parser')
var cors = require('cors')
const webrtc = require('wrtc')

let senderStream = {} // key: topic, value: MediaStream
// NOTE: There is something wrong with the IPs being passed in being
// all or mostly the same. As a result, in essense this only allows
// one publisher or subscriber at a time.
let publishPeers = {} // key: IP4:topic, value: RTCPeerConnection
let subscribePeers = {} // key: IP4:topic, value: RTCPeerConnection

app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true }))
app.use(cors())

const debug = false

app.post('/subscribe', async ({ body }, res) => {
  try {
    console.log(Date(Date.now()).toString(), 'subscribe: got POST on IP', body.ip, 'for topic', body.topic)

    // Configure the peer connection
    const peer = new webrtc.RTCPeerConnection({
      iceServers: [
        {
          urls: 'stun:stun1.l.google.com:19302'
        }
      ]
    })
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: created peer connection object")
    }

    // Close any old peers on the same IP address
    const topic = body.topic
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: got topic", topic)
    }
    const key = body.ip + ':' + topic
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: got key", key)
    }
    if (key in subscribePeers && subscribePeers[key] && subscribePeers[key].connectionState !== 'closed') {
      if (debug) {
        console.log(Date(Date.now()).toString(), "subscribe: peer for key already exists")
      }
      const senders = subscribePeers[key].getSenders()
      if (debug) {
        console.log(Date(Date.now()).toString(), "subscribe: got senders for peer for key")
      }
      senders.forEach((sender) => subscribePeers[key].removeTrack(sender))
      if (debug) {
        console.log(Date(Date.now()).toString(), "subscribe: removed tracks")
      }
      subscribePeers[key].close()
      if (debug) {
        console.log(Date(Date.now()).toString(), "subscribe: closed peer connection")
      }
    }
    subscribePeers[key] = peer
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: set new peer connection")
    }

    const desc = new webrtc.RTCSessionDescription(body.sdp)
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: created desc")
    }
    await peer.setRemoteDescription(desc)
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: set remote desc")
    }

    // Add the publisher's video stream to the subscriber's peer connection
    if (topic in senderStream) {
      if (debug) {
        console.log(Date(Date.now()).toString(), "subscribe: adding topics from publisher")
      }
      senderStream[topic].getTracks().forEach((track) => peer.addTrack(track, senderStream[topic]))
      if (debug) {
        console.log(Date(Date.now()).toString(), "subscribe: added topics from publisher")
      }
    }

    // Create an answer to the publisher's offer
    const answer = await peer.createAnswer()
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: created answer")
    }
    await peer.setLocalDescription(answer)
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: set local description")
    }
    const payload = {
      sdp: answer
    }
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: created payload")
    }

    // Send the answer to the publisher
    res.json(payload)
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: set payload")
    }
  } catch (err) {
    console.error(Date(Date.now()).toString(), 'subscribe: Failed to process subscriber, exception: ' + err.message)
    res.sendStatus(500)
    if (debug) {
      console.log(Date(Date.now()).toString(), "subscribe: sent error status 500")
    }
  }
})

app.post('/publish', async ({ body }, res) => {
  try {
    console.log(Date(Date.now()).toString(), 'publish: got POST on IP', body.ip, 'for topic', body.topic)

    // Configure the peer connection
    const peer = new webrtc.RTCPeerConnection({
      iceServers: [
        {
          urls: 'stun:stun1.l.google.com:19302'
        }
      ]
    })
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: create peer")
    }

    // Close any old peers on the same IP address
    const topic = body.topic
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: got topic", topic)
    }
    const key = body.ip + ':' + topic
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: got key", key)
    }
    if (key in publishPeers && publishPeers[key] && publishPeers[key].connectionState !== 'closed') {
      if (debug) {
        console.log(Date(Date.now()).toString(), "publish: found existing publisher for key", key)
      }
      const senders = publishPeers[key].getSenders()
      if (debug) {
        console.log(Date(Date.now()).toString(), "publish: got senders for old key")
      }
      senders.forEach((sender) => publishPeers[key].removeTrack(sender))
      if (debug) {
        console.log(Date(Date.now()).toString(), "publish: removed tracks for old key")
      }
      publishPeers[key].close()
      if (debug) {
        console.log(Date(Date.now()).toString(), "publish: closed old peer connection")
      }
    }
    publishPeers[key] = peer
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: added new peer")
    }

    // Send the publisher's video stream to all subscribers on that topic
    peer.ontrack = (e) => handleTrackEvent(e, topic)
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: handled track events")
    }

    // Create an answer to the publisher's offer
    const desc = new webrtc.RTCSessionDescription(body.sdp)
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: got desc")
    }
    await peer.setRemoteDescription(desc)
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: set remote description")
    }
    const answer = await peer.createAnswer()
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: got answer")
    }
    await peer.setLocalDescription(answer)
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: set local description")
    }
    const payload = {
      sdp: answer
    }
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: created payload")
    }

    // Send the answer to the publisher
    res.json(payload)
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: set json payload")
    }
  } catch (err) {
    console.error(Date(Date.now()).toString(), 'publish: Failed to process publisher, exception: ' + err.message)
    res.sendStatus(500)
    if (debug) {
      console.log(Date(Date.now()).toString(), "publish: sent error status 500")
    }
  }
})

function handleTrackEvent(e, topic) {
  console.log(Date(Date.now()).toString(), 'Handle track for publisher')
  senderStream[topic] = e.streams[0]
}

app.listen(process.env.REACT_APP_SIGNALLING_SERVER_PORT, () => console.log(Date(Date.now()).toString(), 'Server started'))
