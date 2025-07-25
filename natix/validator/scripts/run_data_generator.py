import argparse
import time
import os
import psutil
import torch

import bittensor as bt

from natix.synthetic_data_generation import SyntheticDataGenerator
from natix.validator.cache import ImageCache
from natix.validator.config import MODEL_NAMES, ROADWORK_IMAGE_CACHE_DIR, SYNTH_CACHE_DIR, get_task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-cache-dir",
        type=str,
        default=ROADWORK_IMAGE_CACHE_DIR,
        help="Directory containing real images to use as reference",
    )
    parser.add_argument("--output-dir", type=str, default=SYNTH_CACHE_DIR, help="Directory to save generated data")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run generation on (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of images to generate per batch")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=MODEL_NAMES,
        help="Specific model to test. If not specified, uses random models",
    )
    args = parser.parse_args()

    if args.model:
        bt.logging.info(f"Using model {args.model} ({get_task(args.model)})")
    else:
        bt.logging.info("No model selected.")

    bt.logging.set_info()

    image_cache = ImageCache(args.image_cache_dir)
    while True:
        if image_cache._extracted_cache_empty():
            bt.logging.info("SyntheticDataGenerator waiting for real image cache to populate")
            time.sleep(5)
            continue
        bt.logging.info("Image cache was populated! Proceeding to data generation")
        break

    sdg = SyntheticDataGenerator(
        prompt_type="annotation",
        use_random_model=args.model is None,
        model_name=args.model,
        device=args.device,
        image_cache=image_cache,
        output_dir=args.output_dir,
    )

    bt.logging.info("Starting data generator service")
    sdg.batch_generate(batch_size=1)

    # Get process for memory monitoring
    process = psutil.Process(os.getpid())
    batch_count = 0

    while True:
        try:
            batch_count += 1
            
            # Monitor memory before generation
            ram_gb = process.memory_info().rss / 1024**3
            bt.logging.info(f"Batch {batch_count} - RAM: {ram_gb:.2f}GB")
            
            # Monitor GPU memory if available
            if torch.cuda.is_available():
                vram_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                vram_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                bt.logging.info(f"Batch {batch_count} - VRAM Allocated: {vram_allocated_gb:.2f}GB, Reserved: {vram_reserved_gb:.2f}GB")
            
            # Run batch generation
            sdg.batch_generate(batch_size=args.batch_size)
            
            # Monitor memory after generation
            ram_gb_after = process.memory_info().rss / 1024**3
            ram_delta = ram_gb_after - ram_gb
            bt.logging.info(f"After batch {batch_count} - RAM: {ram_gb_after:.2f}GB (delta: {ram_delta:+.2f}GB)")
            
            if torch.cuda.is_available():
                vram_allocated_gb_after = torch.cuda.memory_allocated() / 1024**3
                vram_reserved_gb_after = torch.cuda.memory_reserved() / 1024**3
                vram_delta = vram_allocated_gb_after - vram_allocated_gb
                bt.logging.info(f"After batch {batch_count} - VRAM Allocated: {vram_allocated_gb_after:.2f}GB (delta: {vram_delta:+.2f}GB), Reserved: {vram_reserved_gb_after:.2f}GB")
                
        except Exception as e:
            bt.logging.error(f"Error in batch generation: {str(e)}")
            bt.logging.error(f"Traceback: {e.__class__.__name__}: {str(e)}")
            time.sleep(5)
