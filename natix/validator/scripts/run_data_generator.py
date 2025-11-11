import argparse
import time
import os
import psutil
import torch
from pathlib import Path

import bittensor as bt
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from natix.synthetic_data_generation import SyntheticDataGenerator
from natix.validator.cache import ImageCache
from natix.validator.config import MODEL_NAMES, ROADWORK_IMAGE_CACHE_DIR, SYNTH_CACHE_DIR, get_task


def cleanup_old_synthetic_images(output_dir, max_age_days):
    """Remove synthetic images older than max_age_days."""
    current_time = time.time()
    max_age_seconds = max_age_days * 86400  # days to seconds
    removed_count = 0

    output_path = Path(output_dir)
    for task_dir in ["t2i", "i2i"]:
        task_path = output_path / task_dir
        if not task_path.exists():
            continue

        for file_path in task_path.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        # Remove image and its metadata
                        if file_path.suffix in [".png", ".jpg", ".jpeg", ".mp4"]:
                            json_file = file_path.with_suffix(".json")
                            if json_file.exists():
                                json_file.unlink()
                            file_path.unlink()
                            removed_count += 1
                    except Exception as e:
                        bt.logging.warning(f"Failed to remove old file {file_path}: {e}")

    if removed_count > 0:
        bt.logging.info(f"Cleaned up {removed_count} synthetic images older than {max_age_days} days")
    return removed_count


def count_recent_images(output_dir, days=7):
    """Count images generated in the last N days."""
    current_time = time.time()
    cutoff_time = current_time - (days * 86400)  # days to seconds
    count_by_task = {"t2i": 0, "i2i": 0}

    output_path = Path(output_dir)
    for task_dir in ["t2i", "i2i"]:
        task_path = output_path / task_dir
        if not task_path.exists():
            continue

        for file_path in task_path.iterdir():
            if file_path.is_file() and file_path.suffix in [".png", ".jpg", ".jpeg"]:
                if file_path.stat().st_mtime > cutoff_time:
                    count_by_task[task_dir] += 1

    return count_by_task


def run_weekly_generation(args, sdg, image_cache):
    """Run one cycle of weekly generation."""
    bt.logging.info("=" * 80)
    bt.logging.info("Starting weekly generation cycle")
    bt.logging.info("=" * 80)

    # Get process for memory monitoring
    process = psutil.Process(os.getpid())

    # Cleanup old images first
    bt.logging.info(f"Cleaning up images older than {args.max_age_days} days")
    cleanup_old_synthetic_images(args.output_dir, args.max_age_days)

    # Check how many images were generated in the last 7 days
    recent_counts = count_recent_images(args.output_dir, days=7)
    bt.logging.info(f"Images generated in last 7 days - t2i: {recent_counts['t2i']}, i2i: {recent_counts['i2i']}")

    # Calculate how many images we need for each model type
    if args.model:
        # If specific model is set, generate for that model only
        task = get_task(args.model)
        target_remaining = max(0, args.weekly_target - recent_counts.get(task, 0))
        bt.logging.info(f"Target remaining for {task}: {target_remaining} images")
    else:
        # Generate for both t2i and i2i to reach weekly target
        t2i_remaining = max(0, args.weekly_target - recent_counts['t2i'])
        i2i_remaining = max(0, args.weekly_target - recent_counts['i2i'])
        target_remaining = t2i_remaining + i2i_remaining
        bt.logging.info(f"Target remaining - t2i: {t2i_remaining}, i2i: {i2i_remaining} (total: {target_remaining})")

    if target_remaining == 0:
        bt.logging.info("Weekly target already met, no generation needed")
        return

    # Calculate number of batches needed
    num_batches = (target_remaining + args.batch_size - 1) // args.batch_size
    bt.logging.info(f"Generating {num_batches} batches ({target_remaining} images) to reach weekly target")
    bt.logging.info(f"Estimated time on L4: {(target_remaining * 20) / 3600:.1f} hours")

    generation_start_time = time.time()

    try:
        for batch_num in range(num_batches):
            # Monitor memory before generation
            ram_gb = process.memory_info().rss / 1024**3
            bt.logging.info(f"Batch {batch_num + 1}/{num_batches} - RAM: {ram_gb:.2f}GB")

            # Monitor GPU memory if available
            if torch.cuda.is_available():
                vram_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                vram_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                bt.logging.info(f"Batch {batch_num + 1}/{num_batches} - VRAM Allocated: {vram_allocated_gb:.2f}GB, Reserved: {vram_reserved_gb:.2f}GB")

            # Run batch generation
            sdg.batch_generate(batch_size=args.batch_size)

            # Monitor memory after generation
            ram_gb_after = process.memory_info().rss / 1024**3
            ram_delta = ram_gb_after - ram_gb
            bt.logging.info(f"After batch {batch_num + 1}/{num_batches} - RAM: {ram_gb_after:.2f}GB (delta: {ram_delta:+.2f}GB)")

            if torch.cuda.is_available():
                vram_allocated_gb_after = torch.cuda.memory_allocated() / 1024**3
                vram_reserved_gb_after = torch.cuda.memory_reserved() / 1024**3
                vram_delta = vram_allocated_gb_after - vram_allocated_gb
                bt.logging.info(f"After batch {batch_num + 1}/{num_batches} - VRAM Allocated: {vram_allocated_gb_after:.2f}GB (delta: {vram_delta:+.2f}GB), Reserved: {vram_reserved_gb_after:.2f}GB")

        generation_time = time.time() - generation_start_time
        bt.logging.success(f"Completed weekly generation target in {generation_time / 3600:.2f} hours")

    except Exception as e:
        bt.logging.error(f"Error in batch generation: {str(e)}")
        bt.logging.error(f"Traceback: {e.__class__.__name__}: {str(e)}")


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
    parser.add_argument(
        "--weekly-target",
        type=int,
        default=720,
        help="Target number of images to generate per model per week (default: 720)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=10,
        help="Maximum age of synthetic images in days before cleanup (default: 10 days)",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run generation once and exit (for testing or manual runs)",
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

    # Run-once mode for testing
    if args.run_once:
        bt.logging.info("Running in single-run mode (--run-once)")
        bt.logging.info(f"Weekly target: {args.weekly_target} images per model")
        bt.logging.info(f"Max age: {args.max_age_days} days")
        try:
            run_weekly_generation(args, sdg, image_cache)
            bt.logging.info("Single run completed successfully")
        except Exception as e:
            bt.logging.error(f"Error in generation: {e}")
            exit(1)
        exit(0)

    # Normal scheduled mode
    bt.logging.info("Starting weekly data generator service with APScheduler")
    bt.logging.info(f"Scheduled to run every Sunday at 2:00 AM")
    bt.logging.info(f"Weekly target: {args.weekly_target} images per model")
    bt.logging.info(f"Max age: {args.max_age_days} days")

    # Create scheduler
    scheduler = BlockingScheduler()

    # Schedule weekly generation (every Sunday at 2 AM)
    scheduler.add_job(
        lambda: run_weekly_generation(args, sdg, image_cache),
        trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
        id='weekly_generation',
        name='Weekly Synthetic Image Generation',
        max_instances=1,
        replace_existing=True
    )

    bt.logging.info("Scheduler configured successfully")
    bt.logging.info("Running initial generation cycle immediately...")

    # Run once immediately on startup
    try:
        run_weekly_generation(args, sdg, image_cache)
    except Exception as e:
        bt.logging.error(f"Error in initial generation: {e}")

    bt.logging.info("Starting scheduler loop (press Ctrl+C to exit)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        bt.logging.info("Shutting down scheduler...")
        scheduler.shutdown()
        bt.logging.info("Scheduler stopped")
