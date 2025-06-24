import asyncio
import base64
import hashlib
import os
import random
import socket
import threading
import time
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

import bittensor as bt
from httpx import HTTPStatusError, Client, Timeout
import numpy as np
import uvicorn
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import Depends, FastAPI, HTTPException, Request
from PIL import Image

from natix.protocol import prepare_synapse
from natix.utils.image_transforms import get_base_transforms
from natix.utils.uids import get_random_uids
from natix.validator.config import TARGET_IMAGE_SIZE
from natix.validator.proxy import ProxyCounter

base_transforms = get_base_transforms(TARGET_IMAGE_SIZE)


def preprocess_image(b64_image):
    image_bytes = base64.b64decode(b64_image)
    image_buffer = BytesIO(image_bytes)
    pil_image = Image.open(image_buffer)
    return base_transforms(pil_image)


class ValidatorProxy:
    def __init__(
        self,
        validator,
    ):
        self.validator = validator
        try:
            self.get_credentials()
        except Exception as e:
            bt.logging.warning(e)
            bt.logging.warning("Warning, proxy can't ping to proxy-client.")
        self.miner_request_counter = {}
        self.dendrite = bt.dendrite(wallet=validator.wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/validator_proxy",
            self.forward,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_api_route(
            "/health/liveness",
            self.healthcheck,
            methods=["GET"],
            dependencies=[Depends(self.get_self)],
        )

        self.loop = asyncio.get_event_loop()
        self.proxy_counter = ProxyCounter(os.path.join(self.validator.config.neuron.full_path, "proxy_counter.json"))
        
        # Organic task distribution state and configuration
        self.miners_per_task = getattr(self.validator.config.organic, 'miners_per_task', 3)
        self.deduplication_window_seconds = getattr(self.validator.config.organic, 'deduplication_window_seconds', 300)
        self.miner_cooldown_seconds = getattr(self.validator.config.organic, 'miner_cooldown_seconds', 60)
        self.max_concurrent_tasks = getattr(self.validator.config.organic, 'max_concurrent_tasks', 10)
        self.stagger_delay_range = (
            getattr(self.validator.config.organic, 'stagger_delay_min', 0.1),
            getattr(self.validator.config.organic, 'stagger_delay_max', 2.0)
        )
        
        # Organic task tracking state
        self._organic_lock = threading.RLock()
        self._recent_tasks = {}
        self._miner_recent_assignments = defaultdict(lambda: deque(maxlen=100)) 
        self._active_tasks = set() 
        self._completed_tasks = {} 
        
        if self.validator.config.proxy.port:
            self.start_server()

    def get_credentials(self):
        try:
            with Client(timeout=Timeout(30)) as client:
                response = client.post(
                    f"{self.validator.config.proxy.proxy_client_url}/credentials/get",
                    json={
                        "postfix": (
                            f":{self.validator.config.proxy.port}/validator_proxy" if self.validator.config.proxy.port else ""
                        ),
                        "uid": self.validator.uid,
                    },
                )
            response.raise_for_status()
            response = response.json()
            message = response["message"]
            signature = base64.b64decode(response["signature"])

            def verify_credentials(public_key_bytes):
                public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
                try:
                    public_key.verify(signature, message.encode("utf-8"))
                except InvalidSignature:
                    raise Exception("Invalid signature")

            self.verify_credentials = verify_credentials

        except HTTPStatusError as e:
            # Extract and show full response error from server
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail = e.response.text

            bt.logging.warning(f"Credential request failed: {error_detail}")
            bt.logging.warning("Warning, proxy can't ping to proxy-client.")
            return None

        except Exception as e:
            bt.logging.exception(f"Unexpected error while getting credentials: {e}")
            return None
    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(uvicorn.run, self.app, host="0.0.0.0", port=self.validator.config.proxy.port)

    def authenticate_token(self, public_key_bytes):
        public_key_bytes = base64.b64decode(public_key_bytes)
        try:
            self.verify_credentials(public_key_bytes)
            bt.logging.info("Successfully authenticated token")
            return public_key_bytes
        except Exception as e:
            bt.logging.error(f"Exception occurred in authenticating token: {e}")
            bt.logging.error(traceback.print_exc())
            raise HTTPException(status_code=401, detail="Error getting authentication token")

    async def healthcheck(self, request: Request):
        authorization: str = request.headers.get("authorization")

        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        self.authenticate_token(authorization)
        return {"status": "healthy"}

    async def forward(self, request: Request):
        authorization: str = request.headers.get("authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        self.authenticate_token(authorization)

        bt.logging.info("Received an organic request!")
        payload = await request.json()

        if "seed" not in payload:
            payload["seed"] = random.randint(0, int(1e9))

        # Preprocess image
        image = preprocess_image(payload["image"])
        image_bytes = base64.b64decode(payload["image"])
        
        # Prepare synapse
        synapse = prepare_synapse(image, modality="image")
        
        # Use integrated organic task distribution
        additional_params = {"seed": payload["seed"]}
        
        # Add specific miner UIDs if provided in request
        if "miner_uids" in payload:
            additional_params["miner_uids"] = payload["miner_uids"]
        
        task_result = await self._distribute_organic_task(
            image_data=image_bytes,
            synapse=synapse,
            additional_params=additional_params
        )
        
        bt.logging.info(f"[ORGANIC] Task result: {task_result}")
        
        # Handle different task statuses
        if task_result['status'] == 'duplicate':
            bt.logging.info(f"[ORGANIC] Duplicate task {task_result['task_hash']}")
            self.proxy_counter.update(is_success=False)
            self.proxy_counter.save()
            return HTTPException(status_code=429, detail="Duplicate task within time window")
        
        elif task_result['status'] == 'rejected':
            bt.logging.warning(f"[ORGANIC] Task rejected: {task_result.get('reason', 'unknown')}")
            self.proxy_counter.update(is_success=False)
            self.proxy_counter.save()
            return HTTPException(status_code=503, detail=f"Task rejected: {task_result.get('reason', 'unknown')}")
        
        elif task_result['status'] == 'error':
            bt.logging.error(f"[ORGANIC] Task error: {task_result.get('error', 'unknown')}")
            self.proxy_counter.update(is_success=False)
            self.proxy_counter.save()
            return HTTPException(status_code=500, detail="Internal server error")
        
        elif task_result['status'] == 'completed':
            valid_results = task_result['valid_results']
            
            if valid_results:
                self.proxy_counter.update(is_success=True)
                self.proxy_counter.save()
                
                # Extract predictions and UIDs
                valid_preds = [result['result'] for result in valid_results]
                valid_pred_uids = [result['miner_uid'] for result in valid_results]
                
                data = {
                    "preds": [float(p) for p in valid_preds], 
                    "fqdn": socket.getfqdn(),
                    "task_hash": task_result['task_hash'],
                    "miners_queried": task_result['total_miners_queried'],
                    "valid_responses": task_result['valid_responses']
                }

                rich_response: bool = payload.get("rich", "false").lower() == "true"
                if rich_response:
                    metagraph = self.validator.metagraph
                    data["uids"] = [int(uid) for uid in valid_pred_uids]
                    data["ranks"] = [float(metagraph.R[uid]) for uid in valid_pred_uids]
                    data["incentives"] = [float(metagraph.I[uid]) for uid in valid_pred_uids]
                    data["emissions"] = [float(metagraph.E[uid]) for uid in valid_pred_uids]
                    data["hotkeys"] = [str(metagraph.hotkeys[uid]) for uid in valid_pred_uids]
                    data["coldkeys"] = [str(metagraph.coldkeys[uid]) for uid in valid_pred_uids]
                    data["selected_miners"] = task_result['selected_miners']
                    data["distribution_stats"] = self.organic_distributor.get_task_statistics()

                bt.logging.success(f"[ORGANIC] Successfully processed task {task_result['task_hash']}: {len(valid_preds)} valid responses")
                return data
            else:
                self.proxy_counter.update(is_success=False)
                self.proxy_counter.save()
                return HTTPException(status_code=500, detail="No valid responses received")
        
        # Fallback
        self.proxy_counter.update(is_success=False)
        self.proxy_counter.save()
        return HTTPException(status_code=500, detail="Unknown task status")

    # Organic task distribution methods
    async def _distribute_organic_task(
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
        
        with self._organic_lock:
            # Clean up old entries
            self._cleanup_old_entries()
            
            # Generate task hash
            task_hash = self._generate_task_hash(image_data, additional_params)
            
            # Check for duplicate task
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
            
            # Check concurrency limit
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
            
            # Select miners for this task
            # Check if specific miner UIDs are provided
            if additional_params and "miner_uids" in additional_params:
                selected_miners = additional_params["miner_uids"]
                bt.logging.info(f"[ORGANIC] Using specified miner UIDs: {selected_miners}")
            else:
                selected_miners = self._select_miners_for_task(task_hash)
            
            if not selected_miners:
                bt.logging.error(f"[ORGANIC] No available miners for task {task_hash}")
                return {
                    'task_hash': task_hash,
                    'status': 'failed',
                    'reason': 'no_available_miners'
                }
            
            # Record task
            current_time = time.time()
            self._recent_tasks[task_hash] = (current_time, task_hash)
            self._active_tasks.add(task_hash)
            
            bt.logging.info(
                f"[ORGANIC] Distributing task {task_hash} to {len(selected_miners)} miners: {selected_miners}"
            )
        
        try:
            # Prepare task data
            task_data = {
                'task_hash': task_hash,
                'synapse': synapse,
                'selected_miners': selected_miners,
                'timestamp': current_time
            }
            
            # Distribute with staggering
            results = await self._staggered_distribution(selected_miners, task_data)
            
            # Process results
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
            with self._organic_lock:
                self._completed_tasks[task_hash] = task_result
                self._active_tasks.discard(task_hash)
            
            bt.logging.success(
                f"[ORGANIC] Task {task_hash} completed: {len(valid_results)}/{len(selected_miners)} valid responses"
            )
            
            return task_result
            
        except Exception as e:
            bt.logging.error(f"[ORGANIC] Error distributing task {task_hash}: {e}")
            
            with self._organic_lock:
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
            # Sort params for consistent hashing
            sorted_params = sorted(additional_params.items())
            hasher.update(str(sorted_params).encode())
        
        return hasher.hexdigest()[:16]  # Use first 16 chars for readability
    
    def _cleanup_old_entries(self):
        """Clean up old entries from tracking dictionaries."""
        current_time = time.time()
        
        # Clean up recent tasks
        expired_hashes = [
            task_hash for task_hash, (timestamp, _) in self._recent_tasks.items()
            if current_time - timestamp > self.deduplication_window_seconds
        ]
        for task_hash in expired_hashes:
            del self._recent_tasks[task_hash]
        
        # Clean up miner assignments
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
            k=self.validator.metagraph.n.item(),  # Get all available miners
            exclude=exclude_uids or []
        )
        
        # Filter out miners who have been assigned this task recently
        available_miners = []
        for uid in all_available_uids:
            uid = int(uid)  # Ensure it's an int
            recent_assignments = self._miner_recent_assignments[uid]
            
            # Check if miner has been assigned this specific task hash recently
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
        
        # Randomly select miners
        selected_miners = random.sample(available_miners, self.miners_per_task)
        
        # Record assignments
        current_time = time.time()
        for miner_uid in selected_miners:
            self._miner_recent_assignments[miner_uid].append((current_time, task_hash))
        
        return selected_miners
    
    async def _staggered_distribution(self, miners: List[int], task_data: Dict) -> List:
        """Distribute task to miners with random staggering to prevent batch sends."""
        results = []
        
        for i, miner_uid in enumerate(miners):
            # Add random delay except for the first miner
            if i > 0:
                delay = random.uniform(*self.stagger_delay_range)
                await asyncio.sleep(delay)
            
            try:
                # Send task to individual miner
                axon = self.validator.metagraph.axons[miner_uid]
                bt.logging.info(f"[ORGANIC] Sending task {task_data['task_hash']} to miner UID {miner_uid}")
                
                result = await self.dendrite(  # Use self.dendrite (proxy's dendrite)
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

    async def get_self(self):
        return self
