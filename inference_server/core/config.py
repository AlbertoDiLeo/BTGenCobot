"""Configuration for BT Generation Inference Server"""
from pathlib import Path
from pydantic import BaseModel


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False
    log_level: str = "info"


class ModelConfig(BaseModel):
    """Model configuration"""
    # Path to the finetuned Llama 3.2-1B model
    model_path: str = "../models/btgenbot2/models/"

    # Generation parameters
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9

    # Backend configuration
    backend: str = "transformers"  # Hugging Face transformers with Outlines
    device: str = "cuda"  # NVIDIA CUDA (cuda), Apple Metal (mps) or cpu

    # Timeout settings
    generation_timeout: int = 30  # seconds


class PathConfig(BaseModel):
    """Path configuration"""
    # btgenbot2: Llama-3.2-1B fine-tuned model (better semantic quality, faster)
    model_dir: Path = Path(__file__).parent.parent.parent / "models/btgenbot2/models/"
    # No adapter needed - already fine-tuned
    adapter_dir: Path = None
    prompts_dir: Path = Path(__file__).parent.parent / "prompts"

    def __init__(self, **data):
        super().__init__(**data)
        # Resolve paths to absolute
        self.model_dir = self.model_dir.resolve()
        if self.adapter_dir:
            self.adapter_dir = self.adapter_dir.resolve()
        self.prompts_dir = self.prompts_dir.resolve()


# Global configuration instances
server_config = ServerConfig()
model_config = ModelConfig()
path_config = PathConfig()


def get_model_path() -> str:
    """Get the absolute path to the base model directory"""
    return str(path_config.model_dir)


def get_adapter_path() -> str:
    """Get the absolute path to the adapter directory"""
    return str(path_config.adapter_dir) if path_config.adapter_dir else None


def get_generation_params() -> dict:
    """Get default generation parameters"""
    return {
        "max_tokens": model_config.max_tokens,
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
    }
