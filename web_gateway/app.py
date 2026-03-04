from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CommandRequest(BaseModel):
    text: str


class RosPublisher(Node):
    def __init__(self):
        super().__init__("web_gateway_publisher")
        self.pub = self.create_publisher(String, "/btgen_nl_command", 10)


app = FastAPI(title="BTGenCobot Web Gateway", version="0.1")

# CORS (sviluppo locale)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rclpy.init(args=None)
ros_node = RosPublisher()


@app.options("/command")
def options_command():
    return Response(status_code=204)


@app.post("/command")
def post_command(req: CommandRequest):
    msg = String()
    msg.data = req.text
    ros_node.pub.publish(msg)
    return {"ok": True, "published_to": "/btgen_nl_command", "text": req.text}