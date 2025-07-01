import asyncio
import hashlib
import random
import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import bittensor as bt

from natix.utils.uids import get_random_uids


class OrganicTaskDistributor:
    """
    Handles organic task distribution with anti-collusion mechanisms.
    
    This class is designed to be used within an async context and handles
    its own dendrite initialization to avoid context issues.
    """
    
    def __init__(
        self,
        validator,
        miners_per_task: int = 3,
        deduplication_window_seconds: int = 300,
        miner_cooldown_seconds: int = 60,
        max_concurrent_tasks: int = 10,
        stagger_delay_range: Tuple[float, float] = (0.1, 2.0)
    ):
        self.validator = validator
        self.miners_per_task = miners_per_task
        self.deduplication_window_seconds = deduplication_window_seconds
        self.miner_cooldown_seconds = miner_cooldown_seconds
        self.max_concurrent_tasks = max_concurrent_tasks
        self.stagger_delay_range = stagger_delay_range
        
        # State tracking
        self._lock = asyncio.Lock()
        self._recent_tasks = {}
        self._miner_recent_assignments = defaultdict(lambda: deque(maxlen=100))
        self._active_tasks = set()
        self._completed_tasks = {}
        
        # Dendrite will be initialized when first needed in async context
        self._dendrite = None
    
    @property
    def dendrite(self):
        """Lazy initialization of dendrite in async context."""
        if self._dendrite is None:
            if self.validator.config.mock:
                from natix.utils.mock import MockDendrite
                self._dendrite = MockDendrite(wallet=self.validator.wallet)
            else:
                self._dendrite = bt.dendrite(wallet=self.validator.wallet)
            bt.logging.info(f"OrganicTaskDistributor dendrite initialized: {self._dendrite}")
        return self._dendrite
    
    async def distribute_task(
        self, 
        image_data: bytes, 
        synapse, 
        additional_params: Optional[Dict] = None,
        force_new_task: bool = False
    ) -> Dict:
        """
        Distribute an organic task to selected miners with deduplication and staggering.
        
        Args:
            image_data: Raw image bytes for task identification
            synapse: Prepared synapse object for querying miners
            additional_params: Additional parameters for task uniqueness
            force_new_task: If True, bypass deduplication check
            
        Returns:
            Dict containing task results and metadata
        """
        
        async with self._lock:
            self._cleanup_old_entries()

            task_hash = self._generate_task_hash(image_data, additional_params)

            if not force_new_task and self._is_duplicate_task(task_hash):
                existing_timestamp, existing_task_id = self._recent_tasks[task_hash]
                bt.logging.info(
                    f"[ORGANIC] Duplicate task detected {task_hash}, "
                    f"original submitted {time.time() - existing_timestamp:.1f}s ago"
                )
                return {
                    'task_hash': task_hash,
                    'status': 'duplicate',
                    'original_task_id': existing_task_id,
                    'timestamp': existing_timestamp
                }

            if len(self._active_tasks) >= self.max_concurrent_tasks:
                bt.logging.warning(
                    f"[ORGANIC] Maximum concurrent tasks ({self.max_concurrent_tasks}) reached. "
                    f"Rejecting task {task_hash}"
                )
                return {
                    'task_hash': task_hash,
                    'status': 'rejected',
                    'reason': 'max_concurrent_tasks_reached',
                    'active_tasks': len(self._active_tasks)
                }
            
            selected_miners = self._select_miners_for_task(task_hash)
            
            if not selected_miners:
                bt.logging.error(f"[ORGANIC] No available miners for task {task_hash}")
                return {
                    'task_hash': task_hash,
                    'status': 'failed',
                    'reason': 'no_available_miners'
                }

            current_time = time.time()
            self._recent_tasks[task_hash] = (current_time, task_hash)
            self._active_tasks.add(task_hash)
            
            bt.logging.info(
                f"[ORGANIC] Distributing task {task_hash} to {len(selected_miners)} miners: {selected_miners}"
            )
        
        try:
            task_data = {
                'task_hash': task_hash,
                'synapse': synapse,
                'selected_miners': selected_miners,
                'timestamp': current_time
            }

            results = await self._staggered_distribution(selected_miners, task_data)
            
            valid_results = []
            invalid_results = []
            
            for result in results:
                if result.get('result') is not None and result['result'] != -1.0:
                    valid_results.append(result)
                else:
                    invalid_results.append(result)
            
            task_result = {
                'task_hash': task_hash,
                'status': 'completed',
                'selected_miners': selected_miners,
                'valid_results': valid_results,
                'invalid_results': invalid_results,
                'total_miners_queried': len(selected_miners),
                'valid_responses': len(valid_results),
                'timestamp': current_time,
                'completion_time': time.time()
            }
            
            # Store completed task
            async with self._lock:
                self._completed_tasks[task_hash] = task_result
                self._active_tasks.discard(task_hash)
            
            bt.logging.success(
                f"[ORGANIC] Task {task_hash} completed: {len(valid_results)}/{len(selected_miners)} valid responses"
            )
            
            return task_result
            
        except Exception as e:
            bt.logging.error(f"[ORGANIC] Error distributing task {task_hash}: {e}")
            
            async with self._lock:
                self._active_tasks.discard(task_hash)
            
            return {
                'task_hash': task_hash,
                'status': 'error',
                'error': str(e),
                'selected_miners': selected_miners,
                'timestamp': current_time
            }
    
    def _generate_task_hash(self, image_data: bytes, additional_params: Optional[Dict] = None) -> str:
        """Generate a unique hash for the task based on image content and parameters."""
        hasher = hashlib.sha256()
        hasher.update(image_data)
        
        if additional_params:
            sorted_params = sorted(additional_params.items())
            hasher.update(str(sorted_params).encode())
        
        return hasher.hexdigest()[:16]
    
    def _cleanup_old_entries(self):
        """Clean up old entries from tracking dictionaries."""
        current_time = time.time()
        
        expired_hashes = [
            task_hash for task_hash, (timestamp, _) in self._recent_tasks.items()
            if current_time - timestamp > self.deduplication_window_seconds
        ]
        for task_hash in expired_hashes:
            del self._recent_tasks[task_hash]
        
        for miner_uid, assignments in self._miner_recent_assignments.items():
            while assignments and current_time - assignments[0][0] > self.miner_cooldown_seconds:
                assignments.popleft()
    
    def _is_duplicate_task(self, task_hash: str) -> bool:
        """Check if a task is a duplicate within the deduplication window."""
        if task_hash not in self._recent_tasks:
            return False
        
        timestamp, _ = self._recent_tasks[task_hash]
        return time.time() - timestamp < self.deduplication_window_seconds
    
    def _get_available_miners(self, task_hash: str, exclude_uids: Optional[List[int]] = None) -> List[int]:
        """Get miners that haven't been assigned similar tasks recently."""
        current_time = time.time()
        all_available_uids = get_random_uids(
            self.validator, 
            k=self.validator.metagraph.n.item(),
            exclude=exclude_uids or []
        )

        available_miners = []
        for uid in all_available_uids:
            uid = int(uid)
            recent_assignments = self._miner_recent_assignments[uid]

            has_recent_assignment = any(
                task_hash == assigned_hash and current_time - timestamp < self.miner_cooldown_seconds
                for timestamp, assigned_hash in recent_assignments
            )
            
            if not has_recent_assignment:
                available_miners.append(uid)
        
        return available_miners
    
    def _select_miners_for_task(self, task_hash: str, exclude_uids: Optional[List[int]] = None) -> List[int]:
        """Select N random miners for a task, avoiding recent assignments."""
        available_miners = self._get_available_miners(task_hash, exclude_uids)
        
        if len(available_miners) < self.miners_per_task:
            bt.logging.warning(
                f"[ORGANIC] Only {len(available_miners)} miners available for task {task_hash}, "
                f"requested {self.miners_per_task}. Using all available miners."
            )
            return available_miners

        selected_miners = random.sample(available_miners, self.miners_per_task)
        
        current_time = time.time()
        for miner_uid in selected_miners:
            self._miner_recent_assignments[miner_uid].append((current_time, task_hash))
        
        return selected_miners
    
    async def _staggered_distribution(self, miners: List[int], task_data: Dict) -> List:
        """Distribute task to miners with random staggering to prevent batch sends."""
        results = []
        
        for i, miner_uid in enumerate(miners):
            if i > 0:
                delay = random.uniform(*self.stagger_delay_range)
                await asyncio.sleep(delay)
            
            try:
                axon = self.validator.metagraph.axons[miner_uid]
                bt.logging.info(f"[ORGANIC] Sending task {task_data['task_hash']} to miner UID {miner_uid}")
                
                result = await self.dendrite(
                    axons=[axon],
                    synapse=task_data['synapse'],
                    deserialize=True,
                    timeout=9
                )
                
                results.append({
                    'miner_uid': miner_uid,
                    'result': result[0] if result else None,
                    'timestamp': time.time()
                })
                
                bt.logging.success(f"[ORGANIC] Received response from miner UID {miner_uid} for task {task_data['task_hash']}")
                
            except Exception as e:
                bt.logging.error(f"[ORGANIC] Error querying miner UID {miner_uid} for task {task_data['task_hash']}: {e}")
                results.append({
                    'miner_uid': miner_uid,
                    'result': None,
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about task distribution."""
        current_time = time.time()
        
        recent_task_count = sum(
            1 for timestamp, _ in self._recent_tasks.values()
            if current_time - timestamp < self.deduplication_window_seconds
        )

        active_miners = sum(
            1 for assignments in self._miner_recent_assignments.values()
            if assignments and current_time - assignments[-1][0] < self.miner_cooldown_seconds
        )
        
        return {
            'recent_tasks': recent_task_count,
            'active_tasks': len(self._active_tasks),
            'completed_tasks': len(self._completed_tasks),
            'active_miners': active_miners,
            'total_tracked_miners': len(self._miner_recent_assignments),
            'deduplication_window_seconds': self.deduplication_window_seconds,
            'miners_per_task': self.miners_per_task,
            'miner_cooldown_seconds': self.miner_cooldown_seconds
        }