from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

TARGET_IMAGE_SIZE: tuple[int, int] = (224, 224)

MAINNET_UID = 72
TESTNET_UID = 323

# Project constants
MAINNET_WANDB_PROJECT = "natix-subnet"
TESTNET_WANDB_PROJECT = "natix-testnet"
HUGGINGFACE_REPO = "natix-network-org"
WANDB_ENTITY = "natix_network_org"


# Cache directories
HUGGINGFACE_CACHE_DIR: Path = Path.home() / ".cache" / "huggingface"
NATIX_CACHE_DIR: Path = Path.home() / ".cache" / "natix"
NATIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALIDATOR_INFO_PATH: Path = NATIX_CACHE_DIR / "validator.yaml"

ROADWORK_CACHE_DIR: Path = NATIX_CACHE_DIR / "Roadwork"
SYNTH_CACHE_DIR: Path = NATIX_CACHE_DIR / "Synthetic"

ROADWORK_IMAGE_CACHE_DIR: Path = ROADWORK_CACHE_DIR / "image"


T2V_CACHE_DIR: Path = SYNTH_CACHE_DIR / "t2v"
T2I_CACHE_DIR: Path = SYNTH_CACHE_DIR / "t2i"
I2I_CACHE_DIR: Path = SYNTH_CACHE_DIR / "i2i"

# Update intervals in hours
IMAGE_PARQUET_CACHE_UPDATE_INTERVAL = 2
IMAGE_CACHE_UPDATE_INTERVAL = 1

MAX_COMPRESSED_GB = 100
MAX_EXTRACTED_GB = 10

CHALLENGE_TYPE = {0: "None", 1: "Roadwork"}

# Image datasets configuration
IMAGE_DATASETS: Dict[str, List[Dict[str, str]]] = {
    "Roadwork": [
        {"path": f"{HUGGINGFACE_REPO}/roadwork"},
    ],
}


# Prompt generation model configurations
IMAGE_ANNOTATION_MODEL: str = "Salesforce/blip2-opt-2.7b-coco"
TEXT_MODERATION_MODEL: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" if torch.cuda.is_available() else "unsloth/Meta-Llama-3.1-8B-Instruct"

# Text-to-image model configurations
T2I_MODELS: Dict[str, Dict[str, Any]] = {
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {"use_safetensors": True, "torch_dtype": torch.float16, "variant": "fp16"},
        "use_autocast": False,
        "enable_model_cpu_offload": True,
        "vae_enable_slicing": True,
        "vae_enable_tiling": True,
        "generate_args": {
            "guidance_scale": 8.0,
            "num_inference_steps": 25,
            "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu"),
            "negative_prompt": "cartoon, anime, painting, drawing, artistic, stylized, illustration, sketch, unrealistic, fake, artificial, 3d render, cgi, video game, fantasy, sci-fi, aerial view, bird's eye view, satellite view, drone footage, helicopter view, top-down view, security camera, cctv, surveillance camera, indoor scene, interior, portrait, face, person close-up, selfie, vibrant colors, oversaturated, neon, glowing, night vision, thermal imaging, fisheye lens, wide angle distortion, motion blur, blurry, out of focus, unfocused, soft focus, speed blur, camera shake, movement blur",
        },
    },
}
T2I_MODEL_NAMES: List[str] = list(T2I_MODELS.keys())

# Image-to-image model configurations
I2I_MODELS: Dict[str, Dict[str, Any]] = {
    "stabilityai/stable-diffusion-xl-base-1.0-img2img": {
        "pipeline_cls": StableDiffusionXLImg2ImgPipeline,
        "from_pretrained_args": {"model_id": "stabilityai/stable-diffusion-xl-base-1.0", "use_safetensors": True, "torch_dtype": torch.float16, "variant": "fp16"},
        "use_autocast": False,
        "enable_model_cpu_offload": True,
        "vae_enable_slicing": True,
        "vae_enable_tiling": True,
        "generate_args": {
            "guidance_scale": 8.0,
            "num_inference_steps": 25,
            "strength": 0.7,
            "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu"),
            "negative_prompt": "cartoon, anime, painting, drawing, artistic, stylized, illustration, sketch, unrealistic, fake, artificial, 3d render, cgi, video game, fantasy, sci-fi, aerial view, bird's eye view, satellite view, drone footage, helicopter view, top-down view, security camera, cctv, surveillance camera, indoor scene, interior, portrait, face, person close-up, selfie, vibrant colors, oversaturated, neon, glowing, night vision, thermal imaging, fisheye lens, wide angle distortion",
        },
    }
}
I2I_MODEL_NAMES: List[str] = list(I2I_MODELS.keys())

# Combined model configurations
MODELS: Dict[str, Dict[str, Any]] = {**T2I_MODELS, **I2I_MODELS}
MODEL_NAMES: List[str] = list(MODELS.keys())


def get_modality(model_name):
    if model_name in T2I_MODEL_NAMES + I2I_MODEL_NAMES:
        return "image"


def get_task(model_name):
    if model_name in T2I_MODEL_NAMES:
        return "t2i"
    elif model_name in I2I_MODEL_NAMES:
        return "i2i"


def select_random_model(task: Optional[str] = None) -> str:
    """
    Select a random text-to-image

    Args:
        modality: The type of model to select ('t2i', 'i2i', or 'random').
            If None or 'random', randomly chooses between the valid options

    Returns:
        The name of the selected model.

    Raises:
        NotImplementedError: If the specified modality is not supported.
    """
    if task is None or task == "random":
        task = np.random.choice(["t2i", "i2i"])

    if task == "t2i":
        return np.random.choice(T2I_MODEL_NAMES)
    elif task == "i2i":
        return np.random.choice(I2I_MODEL_NAMES)
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
