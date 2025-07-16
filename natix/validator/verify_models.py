import os
import requests
import time
import re
from datetime import datetime
from typing import List

import bittensor as bt
from natix.synthetic_data_generation import SyntheticDataGenerator
from natix.validator.config import IMAGE_ANNOTATION_MODEL, MODEL_NAMES, TEXT_MODERATION_MODEL


def check_miner_model(proxy_client_url: str, miner_uids: List[int]):
    try:
        url = f"{proxy_client_url}/participant/model-validity"
        uid_list = [str(uid) for uid in miner_uids]
        response = requests.post(url, json={"uid_list": uid_list}, timeout=30)
        response.raise_for_status()
        model_validity = response.json()
        return model_validity
    except requests.RequestException as e:
        bt.logging.warning(f"Error fetching model cards: {e}")
        return [False] * len(miner_uids)


def is_model_cached(model_name):
    """
    Check if the specified model is cached by looking for its directory in the Hugging Face cache.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model is cached, False otherwise.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/")
    # Format the directory name correctly by replacing each slash with double dashes
    model_dir = f"models--{model_name.replace('/', '--')}"

    # Construct the full path to where the model directory should be
    model_path = os.path.join(cache_dir, model_dir)

    # Check if the model directory exists
    if os.path.isdir(model_path):
        bt.logging.info(f"{model_name} is in HF cache. Skipping....")
        return True
    else:
        bt.logging.info(f"{model_name} is not cached. Downloading....")
        return False

def main():
    """
    Main function to verify and download validator models.

    This function checks if the required models are cached and downloads them if necessary.
    It also initializes and loads diffusers for uncached models.
    """
    bt.logging.info("Verifying validator model downloads....")
    synthetic_image_generator = SyntheticDataGenerator(prompt_type="annotation", image_cache="test", use_random_model=True)

    # Check and load annotation and moderation models if not cached
    if not is_model_cached(IMAGE_ANNOTATION_MODEL) or not is_model_cached(TEXT_MODERATION_MODEL):
        synthetic_image_generator.prompt_generator.load_models()
        synthetic_image_generator.prompt_generator.clear_gpu()

    # Initialize and load diffusers if not cached
    for model_name in MODEL_NAMES:
        if not is_model_cached(model_name):
            synthetic_image_generator = SyntheticDataGenerator(prompt_type=None, use_random_model=False, model_name=model_name)
            synthetic_image_generator.load_model(model_name)
            synthetic_image_generator.clear_gpu()

if __name__ == "__main__":
    main()
