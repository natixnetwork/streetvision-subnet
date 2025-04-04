import pytest
import numpy as np
import json
from natix.utils.json_utils import make_json_safe, extract_focused_metadata, validate_json

# Test data for the test cases
@pytest.fixture
def test_data():
    return {
        "modality": "image",
        "miner_uids": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "miner_hotkeys": ["key1", "key2", "key3", "key4", "key5"],
        "rewards": np.array([0.1, 0.2, 0.0, 0.4, 0.5], dtype=np.float64),
        "model_urls": ["url1", "", "url3", "", "url5"],
        "miner_image_mcc": [None, None, 0.76, None, 0.82],
        "image": "base64_large_image_data",
        "predictions": [{"complex": "object"}, {"another": "object"}],
        "inf_value": np.inf,
        "numpy_int": np.int64(42),
        "numpy_float": np.float64(3.14)
    }

# Tests for make_json_safe function
def test_make_json_safe_basic_types():
    """Test make_json_safe with basic data types"""
    # Basic types should remain unchanged
    assert make_json_safe("string") == "string"
    assert make_json_safe(123) == 123
    assert make_json_safe(True) is True
    assert make_json_safe(None) is None
    assert make_json_safe(3.14) == 3.14

def test_make_json_safe_numpy_arrays():
    """Test make_json_safe with numpy arrays"""
    # Integer array
    int_array = np.array([1, 2, 3], dtype=np.int64)
    result = make_json_safe(int_array)
    assert isinstance(result, list)
    assert result == [1, 2, 3]
    assert all(isinstance(x, int) for x in result)
    
    # Float array
    float_array = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    result = make_json_safe(float_array)
    assert isinstance(result, list)
    assert result == [1.1, 2.2, 3.3]
    assert all(isinstance(x, float) for x in result)

def test_make_json_safe_numpy_scalars():
    """Test make_json_safe with numpy scalar values"""
    # Integer scalar
    assert make_json_safe(np.int64(42)) == 42
    assert isinstance(make_json_safe(np.int64(42)), int)
    
    # Float scalar
    assert make_json_safe(np.float64(3.14)) == 3.14
    assert isinstance(make_json_safe(np.float64(3.14)), float)

def test_make_json_safe_nested_structures():
    """Test make_json_safe with nested structures"""
    # Nested dictionary with arrays
    nested_dict = {
        "array": np.array([1, 2, 3]),
        "nested": {
            "value": np.float64(3.14),
            "list": [np.int64(1), "string", np.array([4, 5, 6])]
        }
    }
    
    result = make_json_safe(nested_dict)
    assert isinstance(result, dict)
    assert result["array"] == [1, 2, 3]
    assert result["nested"]["value"] == 3.14
    assert result["nested"]["list"][0] == 1
    assert result["nested"]["list"][1] == "string"
    assert result["nested"]["list"][2] == [4, 5, 6]
    
    # Make sure it's JSON serializable
    try:
        json.dumps(result)
        json_ok = True
    except:
        json_ok = False
    assert json_ok

# Tests for extract_focused_metadata function
def test_extract_focused_metadata(test_data):
    """Test extract_focused_metadata with test data"""
    result = extract_focused_metadata(test_data)
    
    # Check that we have the expected keys
    expected_keys = {'modality', 'miner_uids', 'miner_hotkeys', 'rewards', 'model_urls', 'miner_image_mcc'}
    assert set(result.keys()) == expected_keys
    
    # Check that values were converted properly
    assert result["modality"] == "image"
    
    # UIDs should be a list of integers
    assert isinstance(result["miner_uids"], list)
    assert all(isinstance(uid, int) for uid in result["miner_uids"])
    assert result["miner_uids"] == [1, 2, 3, 4, 5]
    
    # Rewards should be a list of floats
    assert isinstance(result["rewards"], list)
    assert all(isinstance(r, float) for r in result["rewards"])
    assert result["rewards"] == [0.1, 0.2, 0.0, 0.4, 0.5]
    
    # MCC values should be preserved
    assert result["miner_image_mcc"] == [None, None, 0.76, None, 0.82]
    
    # Make sure the result is JSON serializable
    try:
        json.dumps(result)
        json_ok = True
    except:
        json_ok = False
    assert json_ok

def test_extract_focused_metadata_empty():
    """Test extract_focused_metadata with empty data"""
    assert extract_focused_metadata({}) == {}
    assert extract_focused_metadata(None) == {}

# Tests for validate_json function
def test_validate_json_valid():
    """Test validate_json with valid data"""
    valid_data = {
        "string": "value",
        "number": 123,
        "list": [1, 2, 3],
        "nested": {"key": "value"}
    }
    
    result = validate_json(valid_data)
    assert result is True

def test_validate_json_invalid():
    """Test validate_json with invalid data"""
    # Create an object that can't be serialized to JSON
    class UnserializableObject:
        def __repr__(self):
            return "UnserializableObject()"
    
    invalid_data = {
        "valid_key": "value",
        "invalid_key": UnserializableObject()
    }
    
    result = validate_json(invalid_data)
    assert result is False

# Integration test - full pipeline
def test_full_pipeline(test_data):
    """Test the full pipeline from extraction to validation"""
    # Extract focused metadata
    focused_data = extract_focused_metadata(test_data)
    
    # Validate it's JSON serializable
    result = validate_json(focused_data)
    assert result is True
    
    # Should be able to serialize to JSON without errors
    try:
        json_str = json.dumps(focused_data)
        json_ok = True
    except:
        json_ok = False
    assert json_ok
    
    # Parse back from JSON and verify key data
    parsed = json.loads(json_str)
    assert parsed["modality"] == "image"
    assert parsed["miner_uids"] == [1, 2, 3, 4, 5]
    assert parsed["rewards"] == [0.1, 0.2, 0.0, 0.4, 0.5]