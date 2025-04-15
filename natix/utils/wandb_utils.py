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
        return [clean_nans_for_json(i) for i in obj] 
    elif isinstance(obj, np.integer):
        return int(obj) 
    elif isinstance(obj, np.floating):
        return float(obj) 
    else:
        return obj

def log_to_wandb(challenge_data, responses, rewards, metrics, scores, axons):
    """Log challenge data to wandb in the same format as the original code"""
    if not wandb.run:
        bt.logging.info("Wandb logging is disabled")
        return
    
    label = challenge_data.get("label")
    modality = challenge_data.get("modality")
    source_model_task = challenge_data.get("source_model_task")
    data_aug_params = challenge_data.get("data_aug_params")
    level = challenge_data.get("data_aug_level")
    miner_uids = challenge_data.get("miner_uids", [])
    
    predictions = [x.prediction for x in responses]
    
    metric_names = set()
    for m in metrics:
        if modality in m and m[modality]: 
            metric_names.update(m[modality].keys())
    
    table_columns = ["miner_uid", "miner_hotkey", "prediction", "reward", "score"]
    for metric_name in sorted(metric_names):
        table_columns.append(metric_name)
    
    miner_table = wandb.Table(columns=table_columns)

    for i, (uid, hotkey, pred, reward, score) in enumerate(zip(
        miner_uids, 
        [axon.hotkey for axon in axons], 
        predictions, 
        rewards, 
        [scores[uid] for uid in miner_uids]
    )):
        row_data = [uid, hotkey, pred, reward, score]
        
        for metric_name in sorted(metric_names):
            metric_value = None
            if i < len(metrics) and modality in metrics[i] and metrics[i][modality]:
                metric_value = metrics[i][modality].get(metric_name, None)
            row_data.append(metric_value)

        miner_table.add_data(*row_data)
    
    metadata_dict = clean_nans_for_json(challenge_data)
    metadata_json = json.dumps(metadata_dict, indent=4)
    try:
        metadata_html = wandb.Html(f"<pre>{metadata_json}</pre>")
    except Exception as e:
        bt.logging.error(f"Unable to create HTML for metadata: {e}")
        metadata_html = None
    
    wandb_log_data = {
        "miner_performance": miner_table,
        "metadata": metadata_html,
    }
    
    for k, v in challenge_data.items():
        if k not in wandb_log_data and not k.startswith(("miner_", "predictions", "rewards", "scores")):
            wandb_log_data[k] = v
    
    try:
        bt.logging.debug("Logging challenge data to wandb")
        wandb.log(wandb_log_data)
    except Exception as e:
        bt.logging.error(f"Error logging to wandb: {e}")