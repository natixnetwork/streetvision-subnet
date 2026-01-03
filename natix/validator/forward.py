# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: dubm
# Copyright © 2023 Natix

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import time
from typing import List
import copy
import ipaddress

import bittensor as bt
import numpy as np
from httpx import HTTPStatusError, Client, Timeout, ReadTimeout

from natix.protocol import prepare_synapse
from natix.utils.image_transforms import apply_augmentation_by_level
from natix.validator.config import CHALLENGE_TYPE, TARGET_IMAGE_SIZE
from natix.validator.reward import get_rewards
from natix.utils.wandb_utils import log_to_wandb

def fix_ip_format(axon):
    ax = copy.copy(axon)
    ip = ax.ip.strip()
    if not (ip.startswith("[") and ip.endswith("]")):
        try:
            if ipaddress.ip_address(ip).version == 6:
                ax.ip = f"[{ip}]"
        except ValueError:
            pass
    return ax

def statistics_assign_task(self, miner_uid_list, type: int, label: int, payload_ref: str):
    """
    Notify the statistics service about an assigned task/challenge.
    This will help us easily have reports on the subnet activity
    such as number of tasks distributed at a period

    Args:
        miner_uid_list (List[int]): UIDs of the miners who received this task.
        label (int): Task label (0: None, 1: Roadwork).
        payload_ref (str): Reference to the task payload (e.g., image string).
    """
    try:
        payload = {
            "validator_uid": int(self.uid),
            "miner_uid_list": [int(uid) for uid in miner_uid_list],
            "type": type,
            "label": int(label),
            "payload_ref": str(payload_ref),
        }

        with Client(timeout=Timeout(30)) as client:
            response = client.post(
                f"{self.config.proxy.proxy_client_url}/organic_tasks/statistics/assign",
                json=payload,
            )

        response.raise_for_status()
        bt.logging.info("Successfully reported task assignment to /statistics/assign")
        return response.json()

    except ReadTimeout:
        bt.logging.warning("Statistics assignment request timed out")
        return None

    except HTTPStatusError as e:
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = e.response.text

        bt.logging.warning(f"Statistics assignment request failed: {error_detail}")
        return None

    except Exception as e:
        bt.logging.exception(f"Unexpected error while assigning task statistics: {e}")
        return None


def statistics_report_task(
    self, miner_uid_list: List[int], predictions: List[float], task_id: str
):
    """
    Notify the statistics service about an responses of task/challenge.
    This will help us gain insights about the overal network behaviour
    for example if the models creating predictions are distorted towards
    a particular response or they are evenly distributed

    Args:
        miner_uid_list (List[int]): UIDs of the miners who received this task.
        predictions (List[float]): Responses received from miners.
        task_id (str): Reference to the task returned by the proxy API.
    """
    try:
        payload = {
            "validator_uid": int(self.uid),
            "miner_uid_list": [int(uid) for uid in miner_uid_list],
            "predictions": [float(prediction) for prediction in predictions],
            "task_id": str(task_id),
        }

        with Client(timeout=Timeout(30)) as client:
            response = client.post(
                f"{self.config.proxy.proxy_client_url}/organic_tasks/statistics/report",
                json=payload,
            )

        response.raise_for_status()
        bt.logging.info("Successfully reported task responses to /statistics/report")
        return response.json()

    except ReadTimeout:
        bt.logging.warning("Statistics report request timed out")
        return None

    except HTTPStatusError as e:
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = e.response.text

        bt.logging.warning(f"Statistics assignment request failed: {error_detail}")
        return None

    except Exception as e:
        bt.logging.exception(f"Unexpected error while assigning task statistics: {e}")
        return None


def determine_challenge_type(media_cache, synthetic_cache, fake_prob=0.5):
    modality = "image"
    label = np.random.choice(list(CHALLENGE_TYPE.keys()))

    use_synthetic = np.random.rand() < fake_prob

    if use_synthetic:
        task = "i2i" if np.random.rand() < 0.5 else "t2i"
        cache = synthetic_cache[modality][task]
        source = "synthetic"
    else:
        cache = media_cache["Roadwork"][modality]
        task = "real"
        source = "real"

    return label, modality, task, cache, source


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.

    Steps are:
    1. Sample miner UIDs
    2. Sample synthetic/real image (50/50 chance for each choice)
    3. Apply random data augmentation to the image
    4. Encode data and prepare Synapse
    5. Query miner axons
    6. Compute rewards and update scores

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    challenge_metadata = {}  # for bookkeeping
    challenge = {}  # for querying miners
    label, modality, source_model_task, cache, source = determine_challenge_type(
        self.media_cache, self.synthetic_media_cache, self._fake_prob
    )
    challenge_metadata["label"] = label
    challenge_metadata["modality"] = modality
    challenge_metadata["source_model_task"] = source_model_task
    challenge_metadata["source"] = source

    bt.logging.info(
        f"Sampling {source} {modality} from {source_model_task if source == 'synthetic' else 'real'} cache"
    )

    if modality != "image":
        bt.logging.error(f"Unexpected modality: {modality}")
        return
    else:
        challenge = cache.sample(label)

    if challenge is None:
        bt.logging.warning("Waiting for cache to populate. Challenge skipped.")
        return

    # update logging dict with everything except image data
    challenge_metadata.update(
        {
            k: v
            for k, v in challenge.items()
            if re.match(r"^(?!image$|video$|videos$|video_\d+$).+", k)
        }
    )
    input_data = challenge[modality]  # extract image

    # apply data augmentation pipeline
    try:
        input_data, level, data_aug_params = apply_augmentation_by_level(
            input_data, TARGET_IMAGE_SIZE, challenge.get("mask_center", None)
        )
    except Exception as e:
        level, data_aug_params = -1, {}
        bt.logging.error(f"Unable to apply augmentations: {e}")

    challenge_metadata["data_aug_params"] = data_aug_params
    challenge_metadata["data_aug_level"] = level

    # sample miner uids for challenge
    miner_uids = self.uid_deck.next_k(
        k=self.config.neuron.sample_size,
        metagraph=self.metagraph,
        vpermit_tao_limit=self.config.neuron.vpermit_tao_limit,
        exclude=None,
    )
    miner_uid_list = miner_uids
    bt.logging.debug(f"Miner UIDs to provide with {source} challenge: {miner_uids}")
    axons = [fix_ip_format(self.metagraph.axons[uid]) for uid in miner_uids]
    challenge_metadata["miner_uids"] = miner_uids
    challenge_metadata["miner_hotkeys"] = list([axon.hotkey for axon in axons])

    # prepare synapse
    synapse = prepare_synapse(input_data, modality=modality)

    try:
        statistics_response = statistics_assign_task(
            self,
            miner_uid_list=miner_uid_list,
            type=0,  # Challenge
            label=int(label),
            payload_ref=synapse.image,
        )
    except Exception as e:
        bt.logging.error(f"Failed to report task assignment to statistics: {e}")

    bt.logging.info(f"Sending {modality} challenge to {len(miner_uids)} miners")
    start = time.time()
    # Here are responses from miners to the challenges (type: 0)
    responses = await self.dendrite(
        axons=axons, synapse=synapse, deserialize=False, timeout=9
    )
    predictions = [x.prediction for x in responses]
    bt.logging.debug(f"Predictions of {source} challenge: {predictions}")

    try:
        statistics_report_task(
            self,
            miner_uid_list=miner_uid_list,
            predictions=predictions,
            task_id=statistics_response["id"],
        )
    except Exception as e:
        bt.logging.error(f"Failed to report task assignment to statistics: {e}")

    bt.logging.info(f"Responses received in {time.time() - start}s")
    bt.logging.success(f"Roadwork {modality} challenge complete!")
    bt.logging.info("Scoring responses")

    rewards, metrics = get_rewards(
        label=label,
        responses=predictions,
        uids=miner_uids,
        axons=axons,
        performance_trackers=self.performance_trackers,
    )

    self.update_scores(rewards, miner_uids)

    for metric_name in list(metrics[0][modality].keys()):
        challenge_metadata[f"miner_{modality}_{metric_name}"] = [
            m[modality][metric_name] for m in metrics
        ]

    challenge_metadata["predictions"] = predictions
    challenge_metadata["rewards"] = rewards.tolist()
    challenge_metadata["scores"] = list(self.scores)

    for uid, pred, reward in zip(miner_uids, predictions, rewards):
        if pred != -1:
            bt.logging.success(f"UID: {uid} | Prediction: {pred} | Reward: {reward}")

    if not self.config.wandb.off:
        log_to_wandb(
            challenge_metadata=challenge_metadata,
            responses=responses,
            rewards=rewards,
            metrics=metrics,
            scores=self.scores,
            axons=axons,
        )

    # ensure state is saved after each challenge
    self.save_miner_history()
    cache._prune_extracted_cache()
