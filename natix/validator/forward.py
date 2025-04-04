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

import bittensor as bt
import numpy as np
import wandb
import json

from natix.protocol import prepare_synapse
from natix.utils.image_transforms import apply_augmentation_by_level
from natix.utils.json_safe import prepare_metadata_for_json
from natix.utils.uids import get_random_uids
from natix.validator.config import CHALLENGE_TYPE, TARGET_IMAGE_SIZE
from natix.validator.reward import get_rewards


def determine_challenge_type(media_cache):
    modality = "image"
    label = np.random.choice(list(CHALLENGE_TYPE.keys()))
    cache = media_cache["Roadwork"][modality]
    task = None
    # if label == 1:
    #     if modality == 'video':
    #         task = 't2v'
    #     elif modality == 'image':
    #         # 20% chance to use i2i (in-painting)
    #         task = 'i2i' if np.random.rand() < 0.2 else 't2i'
    #     cache = cache[task]
    return label, modality, task, cache


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
    label, modality, source_model_task, cache = determine_challenge_type(self.media_cache)
    challenge_metadata["label"] = label
    challenge_metadata["modality"] = modality
    challenge_metadata["source_model_task"] = source_model_task

    bt.logging.info(f"Sampling data from {modality} cache")

    if modality == "video":
        bt.logging.error("Video challenges are not supported")
    elif modality == "image":
        challenge = cache.sample(label)

    if challenge is None:
        bt.logging.warning("Waiting for cache to populate. Challenge skipped.")
        return

    # prepare metadata for logging
    try:
        if modality == "video":
            bt.logging.error("Video challenges are not supported")
        elif modality == "image":
            challenge_metadata["image"] = wandb.Image(challenge["image"])
    except Exception as e:
        bt.logging.error(e)
        bt.logging.error(f"{modality} is truncated or corrupt. Challenge skipped.")
        return

    # update logging dict with everything except image data
    challenge_metadata.update({k: v for k, v in challenge.items() if re.match(r"^(?!image$|video$|videos$|video_\d+$).+", k)})
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
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    bt.logging.debug(f"Miner UIDs: {miner_uids}")
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    challenge_metadata["miner_uids"] = list(miner_uids)
    challenge_metadata["miner_hotkeys"] = list([axon.hotkey for axon in axons])

    bt.logging.debug(f"{input_data}")
    # prepare synapse
    synapse = prepare_synapse(input_data, modality=modality)

    bt.logging.info(f"Sending {modality} challenge to {len(miner_uids)} miners")
    start = time.time()
    responses = await self.dendrite(axons=axons, synapse=synapse, deserialize=False, timeout=9)
    predictions = [x.prediction for x in responses]
    model_urls = [x.model_url for x in responses]
    bt.logging.info(f"Responses received in {time.time() - start}s")
    bt.logging.success(f"Roadwork {modality} challenge complete!")
    bt.logging.info({k: v for k, v in challenge_metadata.items() if k not in ("miner_uids", "miner_hotkeys")})

    bt.logging.info("Scoring responses")
    rewards, metrics = get_rewards(
        label=label, responses=predictions, uids=miner_uids, model_urls=model_urls, axons=axons, performance_trackers=self.performance_trackers
    )

    self.update_scores(rewards, miner_uids)

    metric_names = set()
    for m in metrics:
        if modality in m and m[modality]: 
            metric_names.update(m[modality].keys())

    for metric_name in metric_names:
        challenge_metadata[f"miner_{modality}_{metric_name}"] = [
            m[modality].get(metric_name, None) if modality in m else None 
            for m in metrics
        ]

    challenge_metadata["predictions"] = responses
    challenge_metadata["rewards"] = rewards
    challenge_metadata["scores"] = list(self.scores)
    challenge_metadata["model_urls"] = model_urls

    for uid, pred, reward in zip(miner_uids, predictions, rewards):
        if pred != -1:
            bt.logging.success(f"UID: {uid} | Prediction: {pred} | Reward: {reward}")

    json_safe_challenge_metadata = prepare_metadata_for_json(challenge_metadata)

    try:
        json_test = json.dumps(json_safe_challenge_metadata)
        bt.logging.success("Successfully converted to JSON")
    except Exception as e:
        bt.logging.error(f"JSON conversion error: {e}")

    bt.logging.info(f"Challenge metadata: {json_safe_challenge_metadata}")

    # W&B logging if enabled
    if not self.config.wandb.off:
        wandb.log(challenge_metadata)

    # ensure state is saved after each challenge
    self.save_miner_history()
    cache._prune_extracted_cache()
