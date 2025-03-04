import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import bittensor as bt
import yaml
import wandb
import time
import base64
import requests
from io import BytesIO
from datasets import load_dataset
from PIL import Image

from neurons.validator_proxy import ValidatorProxy
from bitmind.base.validator import BaseValidatorNeuron
from bitmind.validator.config import (
    MAINNET_UID,
    VALIDATOR_INFO_PATH
)
import bitmind
from bitmind.utils.uids import get_random_uids
from bitmind.protocol import ImageSynapse 

HUGGING_REPO = "alirezaght/natix"
WANDB_ENTITY="alirezaght-natix-gmbh"
TESTNET_WANDB_PROJECT="test"
MAINNET_WANDB_PROJECT="test"
class Validator(BaseValidatorNeuron):
    """
    This validator:
    - Selects an image from a dataset.
    - Downloads and encodes the image as base64.
    - Sends the encoded image to miners for classification.
    - Compares miner responses with the ground-truth label.
    - Rewards miners based on accuracy.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("Initializing validator for construction site classification...")

        # Load dataset for challenge selection
        self.dataset = load_dataset(HUGGING_REPO, split="test")
        # Validator proxy for servicing external requests
        self.validator_proxy = ValidatorProxy(self)

        self.init_wandb()
        self.store_vali_info()

    def get_challenge_image(self):
        sample = self.dataset.shuffle().select(range(1))[0]
        image_base64 = sample["image_encoded"]  
        correct_label = sample["label"] 

        return image_base64, correct_label

    def validate_miner_response(self, miner_response, correct_label):
        return 1.0 if miner_response == correct_label else 0.0

    async def forward(self):
        challenge_metadata = {}
        # Get encoded image and correct label
        image_encoded, correct_label = self.get_challenge_image()
        challenge_metadata['label'] = correct_label
        image_bytes = base64.b64decode(image_encoded)
        challenge_metadata['image'] = wandb.Image(Image.open(BytesIO(image_bytes)))
        if image_encoded is None:
            bt.logging.warning("Skipping challenge due to image download failure.")
            return

        # Log challenge
        bt.logging.info(f"Sending challenge | Correct Label: {correct_label}")

        # Select miners to challenge
        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        synapse = ImageSynapse(image=image_encoded)
        
        # Send challenge to miners (includes full image)
        bt.logging.info(f"Querying {len(miner_uids)} miners for classification task...")
        start_time = time.time()
        responses = await self.dendrite(axons=axons, synapse=synapse, deserialize=True, timeout=9)
        bt.logging.info(f"Responses received in {time.time() - start_time:.2f}s")

        # Compare responses with correct label
        rewards = [self.validate_miner_response(response, correct_label) for response in responses]

        # Update miner scores
        self.update_scores(rewards, miner_uids)

        table = wandb.Table(columns=["Miner UID", "Prediction", "Reward", "Score"])
        for uid, pred, reward, score in zip(miner_uids, responses, rewards, self.scores):
            table.add_data(uid, pred, reward, score)

        challenge_metadata['result'] = table
        
        # Log results
        for uid, pred, reward in zip(miner_uids, responses, rewards):
            bt.logging.success(f"UID: {uid} | Prediction: {pred} | Correct Label: {correct_label} | Reward: {reward}")

        if not self.config.wandb.off:
            wandb.log(challenge_metadata)
    
        # Save validator state
        self.save_miner_history()
        

    def init_wandb(self):
        if self.config.wandb.off:
            return

        run_name = f'validator-{self.uid}-{bitmind.__version__}'
        self.config.run_name = run_name
        self.config.uid = self.uid
        self.config.hotkey = self.wallet.hotkey.ss58_address
        self.config.version = bitmind.__version__
        self.config.type = self.neuron_type

        wandb_project = TESTNET_WANDB_PROJECT
        if self.config.netuid == MAINNET_UID:
            wandb_project = MAINNET_WANDB_PROJECT

        bt.logging.info(f"Initializing W&B run for '{WANDB_ENTITY}/{wandb_project}'")
        try:
            run = wandb.init(
                name=run_name,
                project=wandb_project,
                entity=WANDB_ENTITY,
                config=self.config,
                dir=self.config.full_path,
                reinit=True
            )
        except wandb.UsageError as e:
            bt.logging.warning(e)
            bt.logging.warning("Did you run wandb login?")
            return

        signature = self.wallet.hotkey.sign(run.id.encode()).hex()
        self.config.signature = signature
        wandb.config.update(self.config, allow_val_change=True)
        bt.logging.success(f"Started wandb run {run_name}")

    def store_vali_info(self):
        """
        Stores validator details for tracking.
        """
        validator_info = {
            'uid': self.uid,
            'hotkey': self.wallet.hotkey.ss58_address,
            'netuid': self.config.netuid,
            'full_path': self.config.neuron.full_path
        }
        with open(VALIDATOR_INFO_PATH, 'w') as f:
            yaml.safe_dump(validator_info, f, indent=4)

        bt.logging.info(f"Wrote validator info to {VALIDATOR_INFO_PATH}")


# Run the validator
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running | uid {validator.uid} | {time.time()}")
            time.sleep(30)