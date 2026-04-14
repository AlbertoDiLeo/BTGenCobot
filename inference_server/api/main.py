"""FastAPI Server for BehaviorTree Generation"""
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from datetime import datetime
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerState:
    """Global server state"""
    def __init__(self):
        self.start_time = datetime.now()
        self.generator: Any = None
        self.model_loaded = False
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0


state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    logger.info("=" * 80)
    logger.info("Starting BehaviorTree Generation Server")
    logger.info("=" * 80)

    try:
        from core.inference import initialize_generator
        from core.query_rewriter import initialize_rewriter

        logger.info("Loading model...")
        state.generator = initialize_generator()
        state.model_loaded = True
        logger.info("Model loaded successfully!")

        logger.info("Initializing query rewriter...")
        initialize_rewriter()
        logger.info("Query rewriter initialized!")

    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        state.model_loaded = False

    logger.info("Server ready to accept requests")
    logger.info("=" * 80)

    yield

    logger.info("Shutting down server...")


app = FastAPI(
    title="BehaviorTree Generation API",
    description="Generate BehaviorTree XML from natural language commands using MLX-accelerated LLM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_origin_regex=r"^https?://([A-Za-z0-9.-]+)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateBTRequest(BaseModel):
    """Request model for BT generation"""
    command: str = Field(..., description="Natural language command", min_length=1)
    max_tokens: int = Field(1024, description="Maximum tokens to generate", gt=0, le=4096)
    temperature: float = Field(0.6, description="Sampling temperature", ge=0.0, le=2.0)
    prompt_format: str = Field("chat", description="Prompt format: 'chat' (Llama chat) or 'alpaca' (instruction format)")
    use_query_rewriting: bool = Field(True, description="Whether to use LLM query rewriting to expand command")
    custom_instruction: str | None = Field(None, description="Optional custom instruction to override default alpaca_instruction.txt")

    class Config:
        json_schema_extra = {
            "example": {
                "command": "Pick up the red cup and place it on the table",
                "max_tokens": 1024,
                "temperature": 0.6,
                "prompt_format": "alpaca",
                "use_query_rewriting": True
            }
        }


class GenerateBTResponse(BaseModel):
    """Response model for BT generation"""
    bt_xml: Optional[str] = Field(None, description="Generated BehaviorTree XML")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")
    method_used: Optional[str] = Field(None, description="Method used for generation")
    success: bool = Field(..., description="Whether generation succeeded")
    error: Optional[str] = Field(None, description="Error message if generation failed")

    class Config:
        json_schema_extra = {
            "example": {
                "bt_xml": "<?xml version=\"1.0\"?>\n<root BTCPP_format=\"4\">...</root>",
                "generation_time_ms": 1250,
                "method_used": "cfg",
                "success": True,
                "error": None
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Server status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    backend: str = Field("transformers", description="Inference backend")
    uptime_seconds: int = Field(..., description="Server uptime in seconds")
    total_requests: int = Field(0, description="Total number of requests processed")
    successful_requests: int = Field(0, description="Number of successful requests")
    failed_requests: int = Field(0, description="Number of failed requests")


class EmergencyStopRequest(BaseModel):
    """Request model for emergency stop"""
    foxglove_ws_url: str = Field("ws://localhost:8765", description="Foxglove Bridge WebSocket URL")


class EmergencyStopResponse(BaseModel):
    """Response model for emergency stop"""
    success: bool = Field(..., description="Whether stop was successful")
    message: str = Field(..., description="Status message")


class ExecuteCommandRequest(BaseModel):
    """Request model for /execute — generate BT and send command to robot"""
    command: str = Field(..., description="Natural language command", min_length=1)
    temperature: float = Field(0.1, description="Sampling temperature", ge=0.0, le=2.0)
    foxglove_ws_url: str = Field("ws://localhost:8765", description="Foxglove Bridge WebSocket URL")


class ExecuteCommandResponse(BaseModel):
    """Response model for /execute"""
    success: bool = Field(..., description="Whether BT generation succeeded")
    bt_xml: Optional[str] = Field(None, description="Generated BehaviorTree XML")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")
    ros_command_sent: bool = Field(..., description="Whether the command was forwarded to ROS2")
    error: Optional[str] = Field(None, description="Error message if generation failed")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "BehaviorTree Generation API",
        "version": "1.0.0",
        "status": "running" if state.model_loaded else "initializing",
        "endpoints": {
            "POST /generate_bt": "Generate BehaviorTree XML from natural language (no robot execution)",
            "POST /execute": "Generate BT and send command to robot via ROS2",
            "GET /health": "Check server health",
            "POST /emergency_stop": "Emergency stop"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = int((datetime.now() - state.start_time).total_seconds())

    return HealthResponse(
        status="healthy" if state.model_loaded else "unhealthy",
        model_loaded=state.model_loaded,
        backend="transformers",
        uptime_seconds=uptime,
        total_requests=state.total_requests,
        successful_requests=state.successful_requests,
        failed_requests=state.failed_requests
    )


@app.post("/generate_bt", response_model=GenerateBTResponse)
async def generate_bt(request: GenerateBTRequest):
    """
    Generate BehaviorTree XML from natural language command

    This endpoint uses a finetuned Llama 3.2-1B model with MLX acceleration
    and Outlines for structured generation.
    """
    state.total_requests += 1

    if not state.model_loaded or state.generator is None:
        state.failed_requests += 1
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is still initializing."
        )

    logger.info("=" * 80)
    logger.info(f"Received BT generation request")
    logger.info(f"Command: {request.command}")
    logger.info(f"Max tokens: {request.max_tokens}, Temp: {request.temperature}, Format: {request.prompt_format}, Query rewriting: {request.use_query_rewriting}")

    try:
        start_time = time.time()

        # Apply query rewriting if requested
        rewritten_input = None
        if request.use_query_rewriting:
            from core.query_rewriter import rewrite_command
            logger.info("Applying query rewriting...")
            rewritten_input = rewrite_command(request.command)
            if rewritten_input:
                logger.info("=" * 80)
                logger.info("REWRITTEN INPUT:")
                logger.info(rewritten_input)
                logger.info("=" * 80)
            else:
                logger.warning("Query rewriting failed, using original command")

        result = state.generator.generate_bt(
            command=request.command,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            prompt_format=request.prompt_format,
            rewritten_input=rewritten_input,
            custom_instruction=request.custom_instruction
        )

        if result["success"]:
            state.successful_requests += 1
            logger.info(f"✓ Generation successful ({result['method_used']}) in {result['generation_time_ms']}ms")
            logger.info("Generated XML:")
            logger.info(result['bt_xml'])
        else:
            state.failed_requests += 1
            logger.warning(f"✗ Generation failed: {result['error']}")

        logger.info("=" * 80)

        return GenerateBTResponse(**result)

    except Exception as e:
        state.failed_requests += 1
        error_msg = f"Internal server error: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@app.post("/execute", response_model=ExecuteCommandResponse)
async def execute_command(request: ExecuteCommandRequest):
    """
    Generate BehaviorTree from natural language and send the command to the robot.

    This is the main endpoint for the frontend:
    1. Generates the BT XML (with query rewriting via Claude/OpenRouter)
    2. Publishes the command to ROS2 /btgen_nl_command via Foxglove WebSocket Bridge
    3. Returns the generated XML so the frontend can render it immediately

    The robot (inside the Docker container) receives the command, re-generates the BT
    and executes it via Nav2. Real-time execution status is available via Foxglove WS
    topics: /generated_behavior_tree and /behavior_tree_log.
    """
    state.total_requests += 1

    if not state.model_loaded or state.generator is None:
        state.failed_requests += 1
        raise HTTPException(status_code=503, detail="Model not loaded.")

    logger.info("=" * 80)
    logger.info(f"/execute request — Command: {request.command!r}")

    try:
        start_time = time.time()

        # 1. Rewrite the command
        from core.query_rewriter import rewrite_command
        rewritten_input = rewrite_command(request.command)
        if rewritten_input:
            logger.info(f"Rewritten input:\n{rewritten_input}")

        # 2. Generate BT XML (preview for the frontend)
        result = state.generator.generate_bt(
            command=request.command,
            max_tokens=1024,
            temperature=request.temperature,
            prompt_format="alpaca",
            rewritten_input=rewritten_input,
        )

        if not result["success"]:
            state.failed_requests += 1
            return ExecuteCommandResponse(
                success=False,
                bt_xml=None,
                generation_time_ms=result.get("generation_time_ms", 0),
                ros_command_sent=False,
                error=result.get("error"),
            )

        state.successful_requests += 1
        logger.info(f"BT generated in {result['generation_time_ms']}ms")

        # 3. Publish command to ROS2 via Foxglove WebSocket
        from core.ros_bridge import publish_nl_command
        ros_sent = await publish_nl_command(request.command, ws_url=request.foxglove_ws_url)
        if ros_sent:
            logger.info("Command forwarded to ROS2 successfully")
        else:
            logger.warning("Could not forward command to ROS2 (Foxglove Bridge unreachable?)")

        logger.info("=" * 80)

        return ExecuteCommandResponse(
            success=True,
            bt_xml=result["bt_xml"],
            generation_time_ms=result["generation_time_ms"],
            ros_command_sent=ros_sent,
        )

    except Exception as e:
        state.failed_requests += 1
        logger.error(f"/execute error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emergency_stop", response_model=EmergencyStopResponse)
async def emergency_stop(request: EmergencyStopRequest):
    """
    Emergency stop endpoint
    Ferma l'esecuzione corrente inviando il comando 'STOP_EXECUTION' a ROS2,
    senza generare o avviare un nuovo Behavior Tree.
    """
    logger.warning("Emergency stop called, forwarding to ROS2...")

    try:
        from core.ros_bridge import publish_nl_command
        
        # Inviamo la stringa magica "STOP_EXECUTION" che il nodo ROS2 riconosce
        # per cancellare il goal di Nav2 senza chiamare l'LLM
        ros_sent = await publish_nl_command("STOP_EXECUTION", ws_url=request.foxglove_ws_url)
        
        if ros_sent:
            return EmergencyStopResponse(
                success=True,
                message="Execution stopped by user"
            )
        else:
            return EmergencyStopResponse(
                success=False,
                message="Failed to send stop command to ROS2 (Foxglove Bridge unreachable?)"
            )

    except Exception as e:
        logger.error(f"Error during emergency stop: {e}", exc_info=True)
        return EmergencyStopResponse(
            success=False,
            message=f"Internal server error: {str(e)}"
        )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The endpoint {request.url.path} does not exist",
            "available_endpoints": [
                "GET /",
                "GET /health",
                "POST /generate_bt",
                "POST /execute",
                "POST /emergency_stop"
            ]
        }
    )


def main():
    """Main entry point for uv run serve"""
    import uvicorn
    from core.config import server_config

    logger.info("Starting server...")
    logger.info(f"Host: {server_config.host}")
    logger.info(f"Port: {server_config.port}")

    uvicorn.run(
        "api.main:app",
        host=server_config.host,
        port=server_config.port,
        reload=server_config.reload,
        log_level=server_config.log_level
    )


if __name__ == "__main__":
    main()
