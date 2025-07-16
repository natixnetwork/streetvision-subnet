# The MIT License (MIT)
# Copyright © 2023 Yuma
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

import base64
import io
import time
import typing

import bittensor as bt
from PIL import Image

import base_miner.detectors
from base_miner.registry import DETECTOR_REGISTRY
from natix.base.miner import BaseMinerNeuron
from natix.protocol import ExtendedImageSynapse
from natix.utils.config import get_device

class Miner(BaseMinerNeuron):

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_image,
            blacklist_fn=self.blacklist_image,
            priority_fn=self.priority_image,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        bt.logging.info("Loading image detection model if configured")
        self.load_image_detector()

    def load_image_detector(self):
        if (
            str(self.config.neuron.image_detector).lower() == "none"
            or str(self.config.neuron.image_detector_config).lower() == "none"
        ):
            bt.logging.warning("No image detector configuration provided, skipping.")
            self.image_detector = None
            return

        if self.config.neuron.image_detector_device == "auto":
            bt.logging.warning("Automatic device configuration enabled for image detector")
            self.config.neuron.image_detector_device = get_device()

        self.image_detector = DETECTOR_REGISTRY[self.config.neuron.image_detector](
            config_name=self.config.neuron.image_detector_config, device=self.config.neuron.image_detector_device
        )
        bt.logging.info(f"Loaded image detection model: {self.config.neuron.image_detector}")

    async def forward_image(self, synapse: ExtendedImageSynapse) -> ExtendedImageSynapse:
        """
        Perform inference on image

        Args:
            synapse (bt.Synapse): The synapse object containing the list of b64 encoded images in the
            'images' field.

        Returns:
            bt.Synapse: The synapse object with the 'predictions' field populated with a list of probabilities

        """
        if self.image_detector is None:
            bt.logging.info("Image detection model not configured; skipping image challenge")
        else:
            bt.logging.info("Received image challenge!")
            try:
                image_bytes = base64.b64decode(synapse.image)
                image = Image.open(io.BytesIO(image_bytes))
                synapse.prediction = self.image_detector(image)
                synapse.model_url = str(self.config.model_url)

            except Exception as e:
                bt.logging.error("Error performing inference")
                bt.logging.error(e)

            bt.logging.info(f"PREDICTION = {synapse.prediction}")
            label = synapse.testnet_label
            if synapse.testnet_label != -1:
                bt.logging.info(f"LABEL (testnet only) = {label}")
        return synapse

    async def blacklist_image(self, synapse: ExtendedImageSynapse) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_image(self, synapse: ExtendedImageSynapse) -> float:
        return await self.priority(synapse)

    def save_state(self):
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    with Miner() as miner:
        while True:
            log = (
                "Miner | "
                f"UID:{miner.uid} | "
                # f"Block:{self.current_block} | "
                f"Stake:{miner.metagraph.S[miner.uid]:.3f} | "
                f"Trust:{miner.metagraph.T[miner.uid]:.3f} | "
                f"Incentive:{miner.metagraph.I[miner.uid]:.3f} | "
                f"Emission:{miner.metagraph.E[miner.uid]:.3f}"
            )
            bt.logging.info(log)
            time.sleep(5)
