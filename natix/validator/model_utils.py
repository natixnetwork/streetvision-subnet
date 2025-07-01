from typing import Any, Dict, Optional

import bittensor as bt
import torch
from diffusers import MotionAdapter
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_annimatediff_motion_adapter(step: int = 4) -> MotionAdapter:
    """
    Load a motion adapter model for AnimateDiff.

    Args:
        step: The step size for the motion adapter. Options: [1, 2, 4, 8].
        repo: The HuggingFace repository to download the motion adapter from.
        ckpt: The checkpoint filename
    Returns:
        A loaded MotionAdapter model.

    Raises:
        ValueError: If step is not one of [1, 2, 4, 8].
    """
    if step not in [1, 2, 4, 8]:
        raise ValueError("Step must be one of [1, 2, 4, 8]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = MotionAdapter().to(device, torch.float16)

    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    return adapter

def create_pipeline_generator(model_config: Dict[str, Any], model: Any) -> callable:
    """
    Creates a generator function based on pipeline configuration.

    Args:
        model_config: Model configuration dictionary
        model: Loaded model instance(s)

    Returns:
        Callable that handles the generation process for the model
    """
    if isinstance(model_config.get("pipeline_stages"), list):

        def generate(prompt: str, **kwargs):
            output = None
            prompt_embeds = None
            negative_embeds = None

            for stage in model_config["pipeline_stages"]:
                stage_args = {**kwargs}  # Copy base args

                # Add stage-specific args
                if stage.get("input_key") and output is not None:
                    stage_args[stage["input_key"]] = output

                # Add any stage-specific generation args
                if stage.get("args"):
                    stage_args.update(stage["args"])

                # Handle prompt embeddings
                if stage.get("use_prompt_embeds") and prompt_embeds is not None:
                    stage_args["prompt_embeds"] = prompt_embeds
                    stage_args["negative_prompt_embeds"] = negative_embeds
                    stage_args.pop("prompt", None)
                elif stage.get("save_prompt_embeds"):
                    # Get embeddings directly from encode_prompt
                    prompt_embeds, negative_embeds = model[stage["name"]].encode_prompt(
                        prompt=prompt,
                        device=model[stage["name"]].device,
                        num_images_per_prompt=stage_args.get("num_images_per_prompt", 1),
                    )
                    stage_args["prompt_embeds"] = prompt_embeds
                    stage_args["negative_prompt_embeds"] = negative_embeds
                    stage_args.pop("prompt", None)
                else:
                    stage_args["prompt"] = prompt

                # Run stage
                result = model[stage["name"]](**stage_args)

                # Extract output based on stage config
                output = getattr(result, stage.get("output_attr", "images"))

                # Clear memory if configured
                if model_config.get("clear_memory_on_stage_end"):
                    import gc

                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            return result

        return generate

    # Default single-stage pipeline
    return lambda prompt, **kwargs: model(prompt=prompt, **kwargs)

def enable_model_optimizations(
    model: Any,
    device: str,
    enable_cpu_offload: bool = False,
    enable_sequential_cpu_offload: bool = False,
    enable_vae_slicing: bool = False,
    enable_vae_tiling: bool = False,
    disable_progress_bar: bool = True,
    stage_name: Optional[str] = None,
) -> None:
    """
    Enables various model optimizations for better memory usage and performance.

    Args:
        model: The model to optimize
        device: Device to move model to ('cuda', 'cpu', etc)
        enable_cpu_offload: Whether to enable model CPU offloading
        enable_sequential_cpu_offload: Whether to enable sequential CPU offloading
        enable_vae_slicing: Whether to enable VAE slicing
        enable_vae_tiling: Whether to enable VAE tiling
        disable_progress_bar: Whether to disable the progress bar
        stage_name: Optional name of pipeline stage for logging
    """
    model_name = f"{stage_name} " if stage_name else ""

    if disable_progress_bar:
        bt.logging.info(f"Disabling progress bar for {model_name}model")
        model.set_progress_bar_config(disable=True)

    # Handle CPU offloading
    if enable_cpu_offload:
        bt.logging.info(f"Enabling CPU offload for {model_name}model")
        model.enable_model_cpu_offload(device=device)
    elif enable_sequential_cpu_offload:
        bt.logging.info(f"Enabling sequential CPU offload for {model_name}model")
        model.enable_sequential_cpu_offload()
    else:
        # Only move to device if not using CPU offload
        bt.logging.info(f"Moving {model_name}model to {device}")
        model.to(device)

    # Handle VAE optimizations if not using CPU offload
    if not enable_cpu_offload:
        if enable_vae_slicing:
            bt.logging.info(f"Enabling VAE slicing for {model_name}model")
            try:
                model.vae.enable_slicing()
            except Exception:
                try:
                    model.enable_vae_slicing()
                except Exception as e:
                    bt.logging.warning(f"Failed to enable VAE slicing for {model_name}model: {e}")

        if enable_vae_tiling:
            bt.logging.info(f"Enabling VAE tiling for {model_name}model")
            try:
                model.vae.enable_tiling()
            except Exception:
                try:
                    model.enable_vae_tiling()
                except Exception as e:
                    bt.logging.warning(f"Failed to enable VAE tiling for {model_name}model: {e}")
