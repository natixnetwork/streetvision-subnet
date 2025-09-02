import argparse
import time
import os
import psutil
import torch

import bittensor as bt

from natix.synthetic_data_generation import SyntheticDataGenerator
from natix.utils.task_types_client import get_task_types_client
from natix.validator.cache import ImageCache
from natix.validator.config import MODEL_NAMES, get_task, get_real_image_cache_dirs, get_synthetic_cache_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Task type to generate data for (e.g., 'Roadwork', 'Traffic Signs'). If not specified, rotates through all available types.",
    )
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
    
    client = get_task_types_client()
    available_types = client.get_available_challenge_types()
    
    if args.task_type:
        if args.task_type not in available_types:
            bt.logging.error(f"Invalid task type: {args.task_type}. Available types: {available_types}")
            exit(1)
        task_types_to_process = [args.task_type]
    else:
        task_types_to_process = available_types
        bt.logging.info(f"Will rotate through task types: {task_types_to_process}")
    
    real_cache_dirs = get_real_image_cache_dirs()
    synthetic_cache_dirs = get_synthetic_cache_dirs()
    
    generators = {}
    for task_type in task_types_to_process:
        image_cache = ImageCache(real_cache_dirs[task_type])
        
        while True:
            if image_cache._extracted_cache_empty():
                bt.logging.info(f"Waiting for {task_type} real image cache to populate")
                time.sleep(5)
                continue
            bt.logging.info(f"{task_type} image cache was populated!")
            break
        
        output_dir = synthetic_cache_dirs[task_type]["i2i"].parent
        
        generators[task_type] = SyntheticDataGenerator(
            prompt_type="annotation",
            use_random_model=args.model is None,
            model_name=args.model,
            device=args.device,
            image_cache=image_cache,
            output_dir=output_dir,
            task_type=task_type,
        )

    bt.logging.info("Starting data generator service")
    
    for task_type, sdg in generators.items():
        bt.logging.info(f"Initial generation for {task_type}")
        sdg.batch_generate(batch_size=1)

    process = psutil.Process(os.getpid())
    batch_count = 0
    task_index = 0

    while True:
        try:
            batch_count += 1
            current_task = task_types_to_process[task_index % len(task_types_to_process)]
            sdg = generators[current_task]
            
            bt.logging.info(f"\n=== Batch {batch_count} - Task: {current_task} ===")
            
            ram_gb = process.memory_info().rss / 1024**3
            bt.logging.info(f"Batch {batch_count} - RAM: {ram_gb:.2f}GB")
            
            if torch.cuda.is_available():
                vram_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                vram_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                bt.logging.info(f"Batch {batch_count} - VRAM Allocated: {vram_allocated_gb:.2f}GB, Reserved: {vram_reserved_gb:.2f}GB")
            
            sdg.batch_generate(batch_size=args.batch_size)
            
            ram_gb_after = process.memory_info().rss / 1024**3
            ram_delta = ram_gb_after - ram_gb
            bt.logging.info(f"After batch {batch_count} - RAM: {ram_gb_after:.2f}GB (delta: {ram_delta:+.2f}GB)")
            
            if torch.cuda.is_available():
                vram_allocated_gb_after = torch.cuda.memory_allocated() / 1024**3
                vram_reserved_gb_after = torch.cuda.memory_reserved() / 1024**3
                vram_delta = vram_allocated_gb_after - vram_allocated_gb
                bt.logging.info(f"After batch {batch_count} - VRAM Allocated: {vram_allocated_gb_after:.2f}GB (delta: {vram_delta:+.2f}GB), Reserved: {vram_reserved_gb_after:.2f}GB")
            
            task_index += 1
                
        except Exception as e:
            bt.logging.error(f"Error in batch generation: {str(e)}")
            bt.logging.error(f"Traceback: {e.__class__.__name__}: {str(e)}")
            time.sleep(5)
