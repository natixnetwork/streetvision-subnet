import argparse
import asyncio
import os

import bittensor as bt

from natix.validator.cache.image_cache import ImageCache
from natix.validator.config import (
    IMAGE_CACHE_UPDATE_INTERVAL,
    IMAGE_DATASETS,
    IMAGE_PARQUET_CACHE_UPDATE_INTERVAL,
    MAX_COMPRESSED_GB,
    MAX_EXTRACTED_GB,
    NATIX_CACHE_DIR,
    get_available_challenge_types,
)


async def main(args):
    caches = []

    if args.mode in ["all", "image"]:
        bt.logging.info("Starting dynamic image cache updaters")
        
        available_challenges = get_available_challenge_types()
        for challenge_name in available_challenges:
            if challenge_name in IMAGE_DATASETS:
                cache_dir = NATIX_CACHE_DIR / challenge_name.replace(' ', '_').lower() / "image"
                
                challenge_cache = ImageCache(
                    cache_dir=cache_dir,
                    datasets=IMAGE_DATASETS[challenge_name],
                    parquet_update_interval=args.image_parquet_interval,
                    image_update_interval=args.image_interval,
                    num_parquets_per_dataset=5,
                    num_images_per_source=100,
                    max_extracted_size_gb=MAX_EXTRACTED_GB,
                    max_compressed_size_gb=MAX_COMPRESSED_GB,
                )
                challenge_cache.start_updater()
                caches.append(challenge_cache)
                bt.logging.info(f"Started cache updater for {challenge_name}")

    if not caches:
        raise ValueError(f"Invalid mode: {args.mode}")

    while True:
        bt.logging.info(f"Running cache updaters for: {args.mode}")
        await asyncio.sleep(600)  # Status update every 10 minutes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["all", "image"], help="Which cache updater(s) to run")
    parser.add_argument(
        "--image-interval", type=int, default=IMAGE_CACHE_UPDATE_INTERVAL, help="Update interval for images in hours"
    )
    parser.add_argument(
        "--image-parquet-interval",
        type=int,
        default=IMAGE_PARQUET_CACHE_UPDATE_INTERVAL,
        help="Update interval for image parquet files in hours",
    )
    args = parser.parse_args()

    bt.logging.set_info()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        bt.logging.info("Shutting down cache updaters...")
