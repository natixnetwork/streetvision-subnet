#!/usr/bin/env python3
"""
Unit tests for organic task distribution randomness and anti-collusion.

Tests the actual ValidatorProxy distribution methods.
"""

import asyncio
import random
import time
from collections import defaultdict
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest
import sys
import os

# Mock problematic imports before they are loaded
sys.modules['bitsandbytes'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['natix.protocol'] = MagicMock()
sys.modules['natix.utils.image_transforms'] = MagicMock()
sys.modules['natix.validator.config'] = MagicMock()
sys.modules['natix.utils.uids'] = MagicMock()
sys.modules['natix.validator.proxy'] = MagicMock()
sys.modules['bittensor'] = MagicMock()

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock get_random_uids function
def mock_get_random_uids(validator, k, exclude=None):
    all_uids = list(range(50))
    if exclude:
        all_uids = [uid for uid in all_uids if uid not in exclude]
    return all_uids[:k]

# Patch the import
with patch.dict('sys.modules', {
    'natix.utils.uids': MagicMock(get_random_uids=mock_get_random_uids),
    'natix.validator.proxy': MagicMock(ProxyCounter=MagicMock),
    'bittensor': MagicMock(),
    'uvicorn': MagicMock()
}):
    from neurons.validator_proxy import ValidatorProxy


class TestOrganicDistribution:
    """Tests for actual ValidatorProxy organic distribution logic."""
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator for testing."""
        validator = Mock()
        validator.wallet = Mock()
        validator.config = Mock()
        validator.config.neuron.full_path = "/tmp/test"
        validator.config.proxy.port = None
        validator.config.proxy.proxy_client_url = "http://localhost"
        validator.config.organic.miners_per_task = 3
        validator.config.organic.deduplication_window_seconds = 300
        validator.config.organic.miner_cooldown_seconds = 60
        validator.config.organic.max_concurrent_tasks = 10
        validator.config.organic.stagger_delay_min = 0.1
        validator.config.organic.stagger_delay_max = 0.2
        validator.uid = 1
        
        # Mock metagraph
        metagraph = Mock()
        metagraph.n.item.return_value = 50
        metagraph.axons = [Mock() for _ in range(50)]
        validator.metagraph = metagraph
        
        return validator
    
    @pytest.fixture
    def proxy(self, mock_validator):
        """Create ValidatorProxy instance with mocked dependencies."""
        with patch('neurons.validator_proxy.bt.dendrite'), \
             patch('neurons.validator_proxy.get_random_uids', side_effect=mock_get_random_uids), \
             patch('neurons.validator_proxy.ProxyCounter'), \
             patch('neurons.validator_proxy.uvicorn'), \
             patch('neurons.validator_proxy.base64'), \
             patch('neurons.validator_proxy.preprocess_image'):
            
            proxy = ValidatorProxy(mock_validator)
            proxy.dendrite = AsyncMock()
            return proxy
    
    def test_hash_uniqueness(self, proxy):
        """Test that different data produces different hashes."""
        data1 = b"image_data_1"
        data2 = b"image_data_2"
        data3 = b"image_data_3"
        
        hash1 = proxy._generate_task_hash(data1)
        hash2 = proxy._generate_task_hash(data2)
        hash3 = proxy._generate_task_hash(data3)
        
        # All hashes should be different
        assert hash1 != hash2
        assert hash2 != hash3
        assert hash1 != hash3
        
        # Same data should produce same hash
        hash1_repeat = proxy._generate_task_hash(data1)
        assert hash1 == hash1_repeat
    
    def test_hash_collision_resistance(self, proxy):
        """Test that many different inputs produce unique hashes."""
        hashes = set()
        num_tests = 100
        
        for i in range(num_tests):
            data = f"test_data_{i}_{random.randint(0, 1000000)}".encode()
            task_hash = proxy._generate_task_hash(data)
            hashes.add(task_hash)
        
        # Should have no collisions
        assert len(hashes) == num_tests
    
    def test_miner_selection_count(self, proxy):
        """Test that correct number of miners are selected."""
        task_hash = "test_task"
        selected = proxy._select_miners_for_task(task_hash)
        
        assert len(selected) == proxy.miners_per_task
        
        # All selected miners should be unique
        assert len(set(selected)) == len(selected)
    
    def test_miner_selection_randomness(self, proxy):
        """Test that different tasks get different miner combinations."""
        num_tasks = 20
        all_selections = []
        
        for i in range(num_tasks):
            task_hash = f"task_{i}"
            selected = proxy._select_miners_for_task(task_hash)
            all_selections.append(set(selected))
        
        # Check uniqueness
        unique_selections = set(frozenset(selection) for selection in all_selections)
        uniqueness_ratio = len(unique_selections) / len(all_selections)
        
        # Should have good uniqueness (allow some randomness)
        assert uniqueness_ratio >= 0.7, f"Low uniqueness: {uniqueness_ratio:.2%}"
    
    def test_miner_distribution_fairness(self, proxy):
        """Test that miners are selected fairly over many tasks."""
        num_tasks = 60
        miner_usage = defaultdict(int)
        
        for i in range(num_tasks):
            task_hash = f"fairness_test_{i}"
            selected = proxy._select_miners_for_task(task_hash)
            
            for miner_uid in selected:
                miner_usage[miner_uid] += 1
        
        # Check distribution fairness
        if len(miner_usage) > 0:
            usage_counts = list(miner_usage.values())
            mean_usage = sum(usage_counts) / len(usage_counts)
            variance = sum((count - mean_usage) ** 2 for count in usage_counts) / len(usage_counts)
            std_dev = variance ** 0.5
            cv = std_dev / mean_usage if mean_usage > 0 else 0
            
            # Fair distribution should have low coefficient of variation
            assert cv < 0.6, f"Unfair distribution: CV = {cv:.2f}"
            
            # Should use many different miners
            min_miners_expected = proxy.miners_per_task * 3
            assert len(miner_usage) >= min_miners_expected
    
    def test_cooldown_enforcement(self, proxy):
        """Test that miners can't be reassigned the same task immediately."""
        task_hash = "cooldown_test"
        
        # First selection
        selected1 = proxy._select_miners_for_task(task_hash)
        
        # Immediate second selection with same hash
        selected2 = proxy._select_miners_for_task(task_hash)
        
        # Should not overlap due to cooldown
        overlap = set(selected1).intersection(set(selected2))
        assert len(overlap) == 0, f"Cooldown failed: {len(overlap)} miners reused"
    
    def test_different_tasks_different_miners(self, proxy):
        """Test that different task hashes get different miners."""
        task1 = "task_alpha"
        task2 = "task_beta"
        
        selected1 = proxy._select_miners_for_task(task1)
        selected2 = proxy._select_miners_for_task(task2)
        
        # Different tasks should generally get different miners
        # (Allow some overlap since it's random)
        overlap = set(selected1).intersection(set(selected2))
        assert len(overlap) < len(selected1), "All miners were the same for different tasks"
    
    def test_hash_with_params(self, proxy):
        """Test that additional parameters affect hash generation."""
        data = b"same_image_data"
        
        hash1 = proxy._generate_task_hash(data, {"seed": 123})
        hash2 = proxy._generate_task_hash(data, {"seed": 456})
        hash3 = proxy._generate_task_hash(data, None)
        
        # Different params should produce different hashes
        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3
    
    def test_comprehensive_randomness(self, proxy):
        """Comprehensive test of randomness properties."""
        num_tests = 30
        all_selections = []
        all_hashes = []
        
        for i in range(num_tests):
            # Generate unique data
            data = f"comprehensive_test_{i}_{random.randint(0, 1000000)}".encode()
            params = {"seed": random.randint(0, 1000000)}
            
            # Generate hash and select miners
            task_hash = proxy._generate_task_hash(data, params)
            selected_miners = proxy._select_miners_for_task(task_hash)
            
            all_selections.append(set(selected_miners))
            all_hashes.append(task_hash)
        
        # Test hash uniqueness
        unique_hashes = len(set(all_hashes))
        hash_uniqueness = unique_hashes / len(all_hashes)
        assert hash_uniqueness >= 0.95, f"Poor hash uniqueness: {hash_uniqueness:.2%}"
        
        # Test selection uniqueness
        unique_selections = set(frozenset(sel) for sel in all_selections)
        selection_uniqueness = len(unique_selections) / len(all_selections)
        assert selection_uniqueness >= 0.8, f"Poor selection uniqueness: {selection_uniqueness:.2%}"
        
        # Test miner variety
        all_miners_used = set()
        for selection in all_selections:
            all_miners_used.update(selection)
        
        min_expected_miners = proxy.miners_per_task * 2
        assert len(all_miners_used) >= min_expected_miners, f"Too few miners used: {len(all_miners_used)}"
    
    def test_duplicate_detection(self, proxy):
        """Test that duplicate tasks are properly detected."""
        data = b"test_image_data"
        params = {"seed": 12345}
        
        # Generate same hash multiple times
        hash1 = proxy._generate_task_hash(data, params)
        hash2 = proxy._generate_task_hash(data, params)
        
        # Hashes should be identical
        assert hash1 == hash2
        
        # Test duplicate detection in task tracking
        proxy._recent_tasks[hash1] = (time.time(), hash1)
        
        assert proxy._is_duplicate_task(hash1) == True
        assert proxy._is_duplicate_task("different_hash") == False
    
    def test_cleanup_old_entries(self, proxy):
        """Test that old entries are properly cleaned up."""
        # Add old entries
        old_time = time.time() - proxy.deduplication_window_seconds - 10
        proxy._recent_tasks["old_task"] = (old_time, "old_task")
        proxy._recent_tasks["recent_task"] = (time.time(), "recent_task")
        
        # Add old miner assignments
        proxy._miner_recent_assignments[1].append((old_time, "old_task"))
        proxy._miner_recent_assignments[1].append((time.time(), "recent_task"))
        
        # Run cleanup
        proxy._cleanup_old_entries()
        
        # Old task should be removed, recent should remain
        assert "old_task" not in proxy._recent_tasks
        assert "recent_task" in proxy._recent_tasks
        
        # Old assignment should be removed, recent should remain
        assignments = proxy._miner_recent_assignments[1]
        assert len(assignments) == 1
        assert assignments[0][1] == "recent_task"
    
    @pytest.mark.asyncio
    async def test_full_distribution_integration(self, proxy):
        """Test the full _distribute_organic_task method integration."""
        # Mock dendrite responses
        proxy.dendrite.return_value = [0.75]  # Mock prediction result
        
        image_data = b"test_image_for_distribution"
        synapse = Mock()
        params = {"seed": 99999}
        
        # Run full distribution
        result = await proxy._distribute_organic_task(
            image_data=image_data,
            synapse=synapse,
            additional_params=params
        )
        
        # Verify result structure
        assert result['status'] == 'completed'
        assert 'task_hash' in result
        assert 'selected_miners' in result
        assert 'valid_results' in result
        assert 'total_miners_queried' in result
        assert result['total_miners_queried'] == proxy.miners_per_task
        
        # Verify task tracking
        task_hash = result['task_hash']
        assert task_hash in proxy._completed_tasks
        assert task_hash not in proxy._active_tasks
    
    @pytest.mark.asyncio
    async def test_duplicate_task_rejection(self, proxy):
        """Test that duplicate tasks are properly rejected."""
        image_data = b"duplicate_test_image"
        synapse = Mock()
        params = {"seed": 88888}
        
        # Mock dendrite responses
        proxy.dendrite.return_value = [0.5]
        
        # First task should succeed
        result1 = await proxy._distribute_organic_task(
            image_data=image_data,
            synapse=synapse,
            additional_params=params
        )
        assert result1['status'] == 'completed'
        
        # Second identical task should be rejected as duplicate
        result2 = await proxy._distribute_organic_task(
            image_data=image_data,
            synapse=synapse,
            additional_params=params
        )
        assert result2['status'] == 'duplicate'
        assert result2['task_hash'] == result1['task_hash']
    
    @pytest.mark.asyncio
    async def test_concurrent_task_limit(self, proxy):
        """Test that concurrent task limits are enforced."""
        # Set low limit for testing
        proxy.max_concurrent_tasks = 2
        
        # Mock slow dendrite responses
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [0.5]
        
        proxy.dendrite.side_effect = slow_response
        
        # Start multiple tasks concurrently
        tasks = []
        for i in range(5):
            image_data = f"concurrent_test_{i}".encode()
            synapse = Mock()
            task = proxy._distribute_organic_task(
                image_data=image_data,
                synapse=synapse,
                additional_params={"seed": i}
            )
            tasks.append(task)
        
        # Run tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Some should be completed, others rejected due to limit
        completed = [r for r in results if r['status'] == 'completed']
        rejected = [r for r in results if r['status'] == 'rejected']
        
        assert len(completed) <= proxy.max_concurrent_tasks
        assert len(rejected) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])