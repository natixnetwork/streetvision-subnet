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

from natix.protocol import prepare_synapse
from natix.utils.image_transforms import apply_augmentation_by_level
from natix.utils.uids import get_random_uids
from natix.validator.config import CHALLENGE_TYPE, TARGET_IMAGE_SIZE
from natix.validator.reward import get_rewards
from natix.validator.verify_models import check_miner_model
from natix.utils.wandb_utils import log_to_wandb

def determine_challenge_type(media_cache, synthetic_cache, fake_prob=0.5):
    modality = "image"
    label = np.random.choice(list(CHALLENGE_TYPE.keys()))
    
    use_synthetic = np.random.rand() < fake_prob
    
    if use_synthetic:
        task = 'i2i' if np.random.rand() < 0.5 else 't2i'
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

    bt.logging.info(f"Sampling {source} {modality} from {source_model_task if source == 'synthetic' else 'real'} cache")

    if modality != "image":
        bt.logging.error(f"Unexpected modality: {modality}")
        return
    else:
        challenge = cache.sample(label)

    if challenge is None:
        bt.logging.warning("Waiting for cache to populate. Challenge skipped.")
        return
    
    # Log challenge details
    scene_desc = challenge.get("metadata", {}).get("scene_description", "N/A")
    image_path = challenge.get("path", "N/A")
    bt.logging.info(f"Challenge details - Label: {label}, Scene description: {scene_desc}, Image path: {image_path}")

    # try:
    #     if modality == "video":
    #         bt.logging.error("Video challenges are not supported")
    #     elif modality == "image":
    #         # TODO: temporarily disable uploading image to wandb. Takes up a tremendous amount of storage
    #         # Ideally, we could add a URL that points to the image in hugging face
    #         # challenge_metadata["image"] = wandb.Image(challenge["image"])
    # except Exception as e:
    #     bt.logging.error(e)
    #     bt.logging.error(f"{modality} is truncated or corrupt. Challenge skipped.")
    #     return

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
    bt.logging.debug(f"Miner UIDs to provide with {source} challenge: {miner_uids}")
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    challenge_metadata["miner_uids"] = miner_uids.tolist()
    challenge_metadata["miner_hotkeys"] = list([axon.hotkey for axon in axons])

    # prepare synapse
    synapse = prepare_synapse(input_data, modality=modality)

    bt.logging.info(f"Sending {modality} challenge to {len(miner_uids)} miners")
    start = time.time()
    responses = await self.dendrite(axons=axons, synapse=synapse, deserialize=False, timeout=9)
    predictions = [x.prediction for x in responses]
    bt.logging.debug(f"Predictions of {source} challenge: {predictions}")

    # Check model URLs and collect invalid UIDs
    model_urls = [x.model_url for x in responses]
    invalid_uids = set()
    model_validity = check_miner_model(self.config.proxy.proxy_client_url, miner_uids)

    for uid, validity in zip(miner_uids, model_validity):
        if not validity:
            bt.logging.warning(f"Miner UID {uid} missing or invalid model_url.")
            invalid_uids.add(uid)

    bt.logging.info(f"Responses received in {time.time() - start}s")
    bt.logging.success(f"Roadwork {modality} challenge complete!")
    bt.logging.info("Scoring responses")

    # Pass invalid_uids to get_rewards
    rewards, metrics = get_rewards(
        label=label, 
        responses=predictions, 
        uids=miner_uids, 
        axons=axons, 
        performance_trackers=self.performance_trackers,
        invalid_uids=invalid_uids
    )

    self.update_scores(rewards, miner_uids)

    for metric_name in list(metrics[0][modality].keys()):
        challenge_metadata[f"miner_{modality}_{metric_name}"] = [m[modality][metric_name] for m in metrics]

    challenge_metadata["predictions"] = predictions
    challenge_metadata["rewards"] = rewards.tolist()
    challenge_metadata["scores"] = list(self.scores)
    challenge_metadata["model_urls"] = model_urls

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
