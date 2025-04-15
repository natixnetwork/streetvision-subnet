# natix/utils/wandb_utils.py

import json
import numpy as np
import wandb
import bittensor as bt

def clean_nans_for_json(obj):
    """Replace NaN values with None for JSON serialization"""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: clean_nans_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans_for_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return [clean_nans_for_json(i) for i in obj]  # Convert tuples to lists for JSON
    elif isinstance(obj, np.integer):
        return int(obj)  # Convert numpy ints to Python ints
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert numpy floats to Python floats
    else:
        return obj

def metadata_to_html(metadata_dict):
    """Convert metadata dictionary to HTML for wandb display"""
    try:
        # Clean NaN values and other types for JSON serialization
        cleaned_dict = clean_nans_for_json(metadata_dict)
        
        # Convert to JSON with indentation for readability
        metadata_json = json.dumps(cleaned_dict, indent=4)
        
        # Create HTML representation
        metadata_html = wandb.Html(f"<pre>{metadata_json}</pre>")
        return metadata_html
    except Exception as e:
        bt.logging.error(f"Unable to create HTML for metadata: {e}")
        return None

def create_miner_performance_table(challenge_data):
    """Create wandb table for miner performance metrics"""
    miner_uids = challenge_data.get("miner_uids", [])
    axons = challenge_data.get("axons", [])
    predictions = challenge_data.get("predictions", [])
    rewards = challenge_data.get("rewards", [])
    scores = challenge_data.get("scores", {})
    metrics = challenge_data.get("metrics", [])
    modality = challenge_data.get("modality", "")
    
    # Get all metric names from metrics
    metric_names = set()
    for m in metrics:
        if modality in m and m[modality]: 
            metric_names.update(m[modality].keys())
            
    # Define table columns
    table_columns = ["miner_uid", "miner_hotkey", "prediction", "reward", "score"]
    for metric_name in sorted(metric_names):
        table_columns.append(metric_name)
    
    # Create table
    miner_table = wandb.Table(columns=table_columns)
    
    # Add data rows
    for i, (uid, axon, pred, reward) in enumerate(zip(
        miner_uids, axons, predictions, rewards
    )):
        row_data = [uid, axon.hotkey, pred, reward, scores.get(uid, 0)]
        
        for metric_name in sorted(metric_names):
            metric_value = None
            if i < len(metrics) and modality in metrics[i] and metrics[i][modality]:
                metric_value = metrics[i][modality].get(metric_name, None)
            row_data.append(metric_value)

        miner_table.add_data(*row_data)
    
    return miner_table

def log_to_wandb(challenge_data):
    """Log all challenge data to wandb"""
    # Don't proceed if wandb is turned off
    if not wandb.run:
        bt.logging.info("Wandb logging is disabled")
        return
    
    bt.logging.debug("Preparing data for wandb logging")
    
    # Create wandb logging dictionary
    wandb_log_data = {}
    
    # Process metadata to HTML if available
    if "metadata" in challenge_data:
        metadata_html = metadata_to_html(challenge_data["metadata"])
        if metadata_html:
            wandb_log_data["metadata"] = metadata_html
    
    # Create miner performance table if we have the required data
    if all(k in challenge_data for k in ["miner_uids", "predictions", "rewards"]):
        miner_table = create_miner_performance_table(challenge_data)
        wandb_log_data["miner_performance"] = miner_table
    
    # Add other fields from challenge_data
    skip_keys = ["metadata", "miner_uids", "miner_hotkeys", "axons", "predictions", "rewards", "scores", "metrics"]
    
    for k, v in challenge_data.items():
        if k not in skip_keys:
            # Convert numpy types to Python types
            if isinstance(v, np.integer):
                wandb_log_data[k] = int(v)
            elif isinstance(v, np.floating):
                wandb_log_data[k] = float(v)
            else:
                wandb_log_data[k] = v
    
    # Log to wandb
    try:
        bt.logging.info("Logging challenge data to wandb")
        wandb.log(wandb_log_data)
    except Exception as e:
        bt.logging.error(f"Error logging to wandb: {e}")