import json
import numpy as np
import wandb
import bittensor as bt

def clean_nans_for_json(obj):
    """Replace NaN values with None for JSON serialization and handle various non-serializable types"""
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return obj.item()
        except:
            pass
    
    if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
        try:
            return obj.tolist()
        except:
            pass
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nans_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans_for_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return [clean_nans_for_json(i) for i in obj]
    elif isinstance(obj, (str, int, bool, type(None))):
        return obj
    else:
        try:
            return str(obj)
        except:
            return None

def log_to_wandb(challenge_metadata, responses, rewards, metrics, scores, axons):
    """Log challenge metadata to wandb in the same format as the original code"""
    if not wandb.run:
        bt.logging.info("Wandb logging is disabled")
        return
    
    label = challenge_metadata.get("label")
    modality = challenge_metadata.get("modality")
    source_model_task = challenge_metadata.get("source_model_task")
    data_aug_params = challenge_metadata.get("data_aug_params")
    level = challenge_metadata.get("data_aug_level")
    miner_uids = challenge_metadata.get("miner_uids", [])
    
    predictions = [x.prediction for x in responses]
    
    # Calculate aggregated statistics
    miner_scores = [scores[uid] for uid in miner_uids]
    
    # Basic aggregates
    aggregated_stats = {
        "total_reward": sum(rewards),
        "avg_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "avg_score": np.mean(miner_scores),
        "std_score": np.std(miner_scores),
        "min_score": np.min(miner_scores),
        "max_score": np.max(miner_scores),
        "num_miners": len(miner_uids)
    }
    
    metric_names = set()
    for m in metrics:
        if modality in m and m[modality]: 
            metric_names.update(m[modality].keys())
    
    for metric_name in ["accuracy", "auc", "f1_score", "mcc", "precision", "recall"]:
        if metric_name in metric_names:
            metric_values = []
            for i, m in enumerate(metrics):
                if modality in m and m[modality] and metric_name in m[modality]:
                    val = m[modality][metric_name]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        metric_values.append(val)
            
            if metric_values:
                aggregated_stats[f"avg_{metric_name}"] = np.mean(metric_values)
                aggregated_stats[f"std_{metric_name}"] = np.std(metric_values)
                aggregated_stats[f"min_{metric_name}"] = np.min(metric_values)
                aggregated_stats[f"max_{metric_name}"] = np.max(metric_values)
    
    metadata_dict = clean_nans_for_json(challenge_metadata)
    metadata_json = json.dumps(metadata_dict, indent=4)
    try:
        metadata_html = wandb.Html(f"<pre>{metadata_json}</pre>")
    except Exception as e:
        bt.logging.error(f"Unable to create HTML for metadata: {e}")
        metadata_html = None
    
    wandb_log_data = {
        "metadata": metadata_html,
    }
    
    for stat_name, stat_value in aggregated_stats.items():
        wandb_log_data[f"aggregated/{stat_name}"] = clean_nans_for_json(stat_value)
    
    for i, (uid, pred, reward, score) in enumerate(zip(miner_uids, predictions, rewards, [scores[uid] for uid in miner_uids])):
        wandb_log_data[f"miner_predictions/uid_{uid}"] = pred
        wandb_log_data[f"miner_rewards/uid_{uid}"] = reward
        wandb_log_data[f"miner_scores/uid_{uid}"] = score

        if i < len(metrics) and modality in metrics[i] and metrics[i][modality]:
            miner_metrics = metrics[i][modality]
            for metric_name in ["accuracy", "auc", "f1_score", "mcc", "precision", "recall"]:
                if metric_name in miner_metrics:
                    metric_value = miner_metrics[metric_name]
                    metric_value = clean_nans_for_json(metric_value)
                    if metric_value is not None:
                        wandb_log_data[f"miner_{metric_name}/uid_{uid}"] = metric_value
    
    numeric_fields = {"label", "data_aug_level"}
    
    for k, v in challenge_metadata.items():
        if k in numeric_fields and k not in wandb_log_data:
            wandb_log_data[k] = v
    
    try:
        bt.logging.debug("Logging challenge data to wandb")
        wandb.log(wandb_log_data)
    except Exception as e:
        bt.logging.error(f"Error logging to wandb: {e}")