import json
from typing import Any, Dict, List

def make_json_safe(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {k: make_json_safe(v) for k, v in obj.__dict__.items()
                if not k.startswith('_')}
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return str(obj)

def prepare_metadata_for_json(metadata: Dict) -> Dict:
    """Prepare challenge metadata for JSON serialization."""
    json_safe_metadata = {}
    
    for key, value in metadata.items():
        # Skip wandb.Image objects or other non-serializable items
        if key == "image" and hasattr(value, "__class__") and "wandb" in str(value.__class__):
            # For wandb.Image objects, we don't include them in JSON
            continue
        
        # Handle predictions which might be custom objects
        if key == "predictions":
            json_safe_metadata[key] = [make_json_safe(pred) for pred in value]
            continue

        json_safe_metadata[key] = make_json_safe(value)
    
    return json_safe_metadata

# Example of how to use:
# Before logging to wandb or other systems that require JSON:
# json_safe_challenge_metadata = prepare_metadata_for_json(challenge_metadata)

# Test serialization:
# try:
#     json_test = json.dumps(json_safe_challenge_metadata)
#     print("Successfully converted to JSON")
# except Exception as e:
#     print(f"JSON conversion error: {e}")

# Then use the safe version for logging or other purposes
# wandb.log(json_safe_challenge_metadata)