import asyncio
import base64
import os
import random
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional

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
from natix.validator.config import TARGET_IMAGE_SIZE
from natix.validator.proxy import ProxyCounter
from natix.validator.organic_task_distributor import OrganicTaskDistributor

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
        
        # Initialize organic task distributor
        self.organic_distributor = OrganicTaskDistributor(
            validator=validator,
            miners_per_task=getattr(self.validator.config.organic, 'miners_per_task', 3),
            deduplication_window_seconds=getattr(self.validator.config.organic, 'deduplication_window_seconds', 300),
            miner_cooldown_seconds=getattr(self.validator.config.organic, 'miner_cooldown_seconds', 60),
            max_concurrent_tasks=getattr(self.validator.config.organic, 'max_concurrent_tasks', 10),
            stagger_delay_range=(
                getattr(self.validator.config.organic, 'stagger_delay_min', 0.1),
                getattr(self.validator.config.organic, 'stagger_delay_max', 2.0)
            )
        ) 
        
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

        image = preprocess_image(payload["image"])
        image_bytes = base64.b64decode(payload["image"])
        synapse = prepare_synapse(image, modality="image")
        additional_params = {"seed": payload["seed"]}
        
        task_result = await self.organic_distributor.distribute_task(
            image_data=image_bytes,
            synapse=synapse,
            additional_params=additional_params
        )
        
        bt.logging.info(f"[ORGANIC] Task result: {task_result}")

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
                    data["distribution_stats"] = self.organic_distributor.get_statistics()

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


    async def get_self(self):
        return self
