import os
import requests
import time

import bittensor as bt
from natix.synthetic_data_generation import SyntheticDataGenerator
from natix.validator.config import IMAGE_ANNOTATION_MODEL, MODEL_NAMES, TEXT_MODERATION_MODEL
from natix.utils.model_format import REQUIRED_MODEL_CARD_KEYS, OPTIONAL_MODEL_CARD_KEYS

def uid_to_hotkey(uid: int, hotkeys: list[str]) -> str:
    return hotkeys[uid]

def is_valid_timestamp(ts: int) -> bool:
    current_time = int(time.time())
    if ts > 10**12:  # If in milliseconds
        ts //= 1000
    return 946684800 <= ts <= current_time


def fetch_model_card(model_url: str, uid: int, version: int = None):
    model_repo = f"{model_url}"
    if version:
        model_repo += f"-v{version}"

    model_card_url = f"https://huggingface.co/{model_repo}/resolve/main/model_card.json"
    try:
        response = requests.get(model_card_url)
        response.raise_for_status()
        model_card = response.json()
    except requests.RequestException as e:
        bt.logging.warning(f"Error fetching model card for {model_repo}: {e}")
        return None

    return model_card

def validate_model_card(card: dict, uid: int, hotkeys: list[str]):
    missing_keys = [k for k in REQUIRED_MODEL_CARD_KEYS if k not in card]
    if missing_keys:
        bt.logging.warning(f"Invalid model card: missing keys {missing_keys}")
        return False

    submitted_by = card['submitted_by']
    expected_hotkey = uid_to_hotkey(uid, hotkeys)

    if submitted_by != expected_hotkey:
        bt.logging.warning(f"Model card submitted_by {submitted_by} does not match UID {uid}'s hotkey {expected_hotkey}")
        return False

    if not is_valid_timestamp(int(card['submission_time'])):
        bt.logging.warning(f"Invalid submission_time: {card['submission_time']}")
        return False

    bt.logging.info(f"The submitted model for UID: {uid} was valid.")
    return True

def check_miner_model(model_url: str, uid: int, hotkeys: list[str]):
    card = fetch_model_card(model_url, uid)
    if not card:
        bt.logging.warning(f"No model found for uid: {uid} with model url: {model_url}")
        return False

    if validate_model_card(card, uid, hotkeys):
        bt.logging.info(f"Model card for uid: {uid} and model url: {model_url} is valid")
        return True
    else:
        bt.logging.warning(f"Invalid model card for uid: {uid} and model url: {model_url}")
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
