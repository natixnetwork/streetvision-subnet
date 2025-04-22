import os
import requests

import bittensor as bt
from natix.synthetic_data_generation import SyntheticDataGenerator
from natix.validator.config import IMAGE_ANNOTATION_MODEL, MODEL_NAMES, TEXT_MODERATION_MODEL
from natix.utils.model_format import REQUIRED_MODEL_CARD_KEYS, OPTIONAL_MODEL_CARD_KEYS

def fetch_model_card(miner_wallet: str, version: int = None):
    model_repo = f"natix-subnet/{miner_wallet}"
    if version:
        model_repo += f"-v{version}"

    model_card_url = f"https://huggingface.co/{model_repo}/resolve/main/model_card.json"

    try:
        response = requests.get(model_card_url)
        response.raise_for_status()
        model_card = response.json()
    except requests.RequestException as e:
        print(f"Error fetching model card for {model_repo}: {e}")
        return None

    return model_card

def validate_model_card(card: dict):
    missing_keys = [k for k in REQUIRED_MODEL_CARD_KEYS if k not in card]
    if missing_keys:
        print(f"Invalid model card: missing keys {missing_keys}")
        return False
    return True

def check_miner_model(miner_wallet: str):
    card = fetch_model_card(miner_wallet)
    if not card:
        print(f"No model found for {miner_wallet}")
        return False

    if validate_model_card(card):
        print(f"Model card for {miner_wallet} is valid")
        return True
    else:
        print(f"Invalid model card for {miner_wallet}")
        return False

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
