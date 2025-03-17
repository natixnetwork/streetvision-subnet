from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from diffusers import (
    AutoPipelineForInpainting,
    FluxPipeline,
    IFPipeline,
    IFSuperResolutionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

TARGET_IMAGE_SIZE: tuple[int, int] = (224, 224)

MAINNET_UID = 34
TESTNET_UID = 168

# Project constants
MAINNET_WANDB_PROJECT: str = "test"
TESTNET_WANDB_PROJECT: str = "test"
HUGGINGFACE_REPO: str = "natix-network-org"
WANDB_ENTITY: str = "alirezaght-natix-gmbh"


# Cache directories
HUGGINGFACE_CACHE_DIR: Path = Path.home() / ".cache" / "huggingface"
NATIX_CACHE_DIR: Path = Path.home() / ".cache" / "natix"
NATIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALIDATOR_INFO_PATH: Path = NATIX_CACHE_DIR / "validator.yaml"

CLEAR_CACHE_DIR: Path = NATIX_CACHE_DIR / "None"
ROADWORK_CACHE_DIR: Path = NATIX_CACHE_DIR / "Roadwork"
SYNTH_CACHE_DIR: Path = NATIX_CACHE_DIR / "Synthetic"

ROADWORK_IMAGE_CACHE_DIR: Path = ROADWORK_CACHE_DIR / "image"
CLEAR_IMAGE_CACHE_DIR: Path = CLEAR_CACHE_DIR / "image"


T2V_CACHE_DIR: Path = SYNTH_CACHE_DIR / "t2v"
T2I_CACHE_DIR: Path = SYNTH_CACHE_DIR / "t2i"
I2I_CACHE_DIR: Path = SYNTH_CACHE_DIR / "i2i"

# Update intervals in hours
IMAGE_PARQUET_CACHE_UPDATE_INTERVAL = 2
IMAGE_CACHE_UPDATE_INTERVAL = 1

MAX_COMPRESSED_GB = 100
MAX_EXTRACTED_GB = 10

CHALLENGE_TYPE = {0: "Clear", 1: "Roadwork"}

# Image datasets configuration
IMAGE_DATASETS: Dict[str, List[Dict[str, str]]] = {
    "Roadwork": [
        {"path": "natix-network-org/roadwork"},
    ],
}

# Prompt generation model configurations
IMAGE_ANNOTATION_MODEL: str = "Salesforce/blip2-opt-6.7b-coco"
TEXT_MODERATION_MODEL: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# Text-to-image model configurations
T2I_MODELS: Dict[str, Dict[str, Any]] = {
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {"use_safetensors": True, "torch_dtype": torch.float16, "variant": "fp16"},
        "use_autocast": False,
    },
    "SG161222/RealVisXL_V4.0": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {"use_safetensors": True, "torch_dtype": torch.float16, "variant": "fp16"},
    },
    "Corcelio/mobius": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {"use_safetensors": True, "torch_dtype": torch.float16},
    },
    "black-forest-labs/FLUX.1-dev": {
        "pipeline_cls": FluxPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16,
        },
        "generate_args": {
            "guidance_scale": 2,
            "num_inference_steps": {"min": 50, "max": 125},
            "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu"),
            "resolution": [512, 768],
        },
        "enable_model_cpu_offload": False,
    },
    "prompthero/openjourney-v4": {
        "pipeline_cls": StableDiffusionPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
        },
    },
    "cagliostrolab/animagine-xl-3.1": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
        },
    },
    "DeepFloyd/IF": {
        "pipeline_cls": {"stage1": IFPipeline, "stage2": IFSuperResolutionPipeline},
        "from_pretrained_args": {
            "stage1": {
                "base": "DeepFloyd/IF-I-XL-v1.0",
                "torch_dtype": torch.float16,
                "variant": "fp16",
                "clean_caption": False,
                "watermarker": None,
                "requires_safety_checker": False,
            },
            "stage2": {
                "base": "DeepFloyd/IF-II-L-v1.0",
                "torch_dtype": torch.float16,
                "variant": "fp16",
                "text_encoder": None,
                "watermarker": None,
                "requires_safety_checker": False,
            },
        },
        "pipeline_stages": [
            {
                "name": "stage1",
                "args": {"output_type": "pt", "num_images_per_prompt": 1, "return_dict": True},
                "output_attr": "images",
                "output_transform": lambda x: x[0].unsqueeze(0),
                "save_prompt_embeds": True,
            },
            {
                "name": "stage2",
                "input_key": "image",
                "args": {"output_type": "pil", "num_images_per_prompt": 1},
                "output_attr": "images",
                "use_prompt_embeds": True,
            },
        ],
        "clear_memory_on_stage_end": True,
    },
}
T2I_MODEL_NAMES: List[str] = list(T2I_MODELS.keys())

# Image-to-image model configurations
I2I_MODELS: Dict[str, Dict[str, Any]] = {
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": {
        "pipeline_cls": AutoPipelineForInpainting,
        "from_pretrained_args": {"use_safetensors": True, "torch_dtype": torch.float16, "variant": "fp16"},
        "generate_args": {
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "strength": 0.99,
            "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu"),
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
    Select a random text-to-image or text-to-video model based on the specified
    modality.

    Args:
        modality: The type of model to select ('t2v', 't2i', 'i2i', or 'random').
            If None or 'random', randomly chooses between the valid options

    Returns:
        The name of the selected model.

    Raises:
        NotImplementedError: If the specified modality is not supported.
    """
    if task is None or task == "random":
        task = np.random.choice(["t2i", "i2i", "t2v"])

    if task == "t2i":
        return np.random.choice(T2I_MODEL_NAMES)
    elif task == "i2i":
        return np.random.choice(I2I_MODEL_NAMES)
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
