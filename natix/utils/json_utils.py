"""
Utility functions for handling JSON serialization with focus on specific metrics.
"""
import json
import numpy as np
import bittensor as bt

def make_json_safe(obj):
    """Convert objects to JSON-serializable format."""
    # NumPy arrays - convert to list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # NumPy scalar types
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        float_val = float(obj)
        if np.isnan(float_val) or np.isinf(float_val):
            return None
        return float_val
    
    # Lists and tuples - process recursively
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    
    # Dictionaries - process recursively
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    
    # Basic types that don't need conversion
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # Default case - convert to string
    else:
        return str(obj)

def extract_focused_metadata(metadata):
    """
    Extract only the key metrics we're interested in.
    
    Args:
        metadata: Original challenge metadata dictionary
        
    Returns:
        Dictionary with only the specified metrics
    """
    # Handle None metadata
    if metadata is None:
        return {}
    
    focused_data = {}
    
    # Core information to keep
    key_fields = [
        'modality',
        'miner_uids',
        'miner_hotkeys',
        'rewards',
        'model_urls'
    ]
    
    # Extract the main fields
    for field in key_fields:
        if field in metadata:
            focused_data[field] = make_json_safe(metadata[field])
    
    # Extract specific metrics (mcc)
    for key in metadata:
        if key.endswith('_mcc'):
            focused_data[key] = make_json_safe(metadata[key])
    
    return focused_data

def validate_json(data):
    """Validate that data can be serialized to JSON."""
    try:
        json.dumps(data)
        bt.logging.info("Successfully serialized data to JSON")
        return True
    except Exception as e:
        bt.logging.error(f"JSON serialization error: {e}")
        
        # Find the problematic keys
        problematic_keys = []
        for key, value in data.items():
            try:
                json.dumps({key: value})
            except:
                problematic_keys.append(key)
                
        bt.logging.error(f"Problematic keys: {problematic_keys}")
        
        return False