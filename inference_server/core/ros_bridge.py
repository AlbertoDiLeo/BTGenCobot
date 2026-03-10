"""Foxglove WebSocket client to publish ROS2 messages from the inference server.

Uses the Foxglove WebSocket Protocol v1 (clientPublish capability) to publish
std_msgs/String messages to /btgen_nl_command without requiring ROS2 on the host.
"""
import asyncio
import json
import logging
import struct

import websockets

logger = logging.getLogger(__name__)

FOXGLOVE_WS_URL = "ws://localhost:8765"
TOPIC = "/btgen_nl_command"
CHANNEL_ID = 1


async def publish_nl_command(command: str, ws_url: str = FOXGLOVE_WS_URL) -> bool:
    """
    Publish a natural language command to the ROS2 /btgen_nl_command topic
    via the Foxglove WebSocket Bridge.

    Uses the Foxglove WebSocket Protocol v1 binary client-publish format:
      [0x01][uint32 channelId LE][uint64 timestamp_ns LE][JSON payload]

    Returns True if the message was sent successfully, False otherwise.
    """
    try:
        async with websockets.connect(
            ws_url,
            subprotocols=["foxglove.sdk.v1"],
            open_timeout=5.0,
            max_size=None,
        ) as ws:
            # 1. Wait for serverInfo handshake
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            info = json.loads(raw)
            if info.get("op") != "serverInfo":
                logger.error(f"Foxglove: expected serverInfo, got '{info.get('op')}'")
                return False

            capabilities = info.get("capabilities", [])
            if "clientPublish" not in capabilities:
                logger.error(f"Foxglove bridge does not support clientPublish. Capabilities: {capabilities}")
                return False

            # 2. Advertise a channel for std_msgs/String with JSON encoding
            await ws.send(json.dumps({
                "op": "advertise",
                "channels": [{
                    "id": CHANNEL_ID,
                    "topic": TOPIC,
                    "encoding": "json",
                    "schemaName": "std_msgs/String",
                    "schema": json.dumps({
                        "type": "object",
                        "properties": {"data": {"type": "string"}}
                    })
                }]
            }))

            # Small delay so the bridge registers the channel before we publish
            await asyncio.sleep(0.15)

            # 3. Send binary message data
            #    foxglove.sdk.v1 format: 0x01 | uint32(channelId) | bytes(payload)
            #    Note: foxglove.sdk.v1 does NOT include a timestamp in the client
            #    publish header (unlike the old foxglove.websocket.v1 protocol).
            payload = json.dumps({"data": command}).encode("utf-8")
            header = struct.pack("<BI", 0x01, CHANNEL_ID)
            await ws.send(header + payload)

            logger.info(f"Published to {TOPIC}: {command!r}")

            # Give the bridge time to forward the message to ROS2
            await asyncio.sleep(0.2)

            # 4. Unadvertise
            await ws.send(json.dumps({
                "op": "unadvertise",
                "channelIds": [CHANNEL_ID]
            }))

            return True

    except asyncio.TimeoutError:
        logger.error(f"Timeout connecting to Foxglove Bridge at {ws_url}")
        return False
    except OSError as e:
        logger.error(f"Cannot reach Foxglove Bridge at {ws_url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to publish ROS command: {e}")
        return False
