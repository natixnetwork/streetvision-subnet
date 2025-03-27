# The MIT License (MIT)
# Copyright © 2025 Natix

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
from io import BytesIO

import bittensor as bt
import pydantic
import torch
from PIL import Image
from torchvision import transforms

from natix.utils.image_transforms import get_base_transforms
from natix.validator.config import TARGET_IMAGE_SIZE

base_transforms = get_base_transforms(TARGET_IMAGE_SIZE)

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   b64_images = [b64_img_1, ..., b64_img_n]
#   predictions = dendrite.query( ImageSynapse( images = b64_images ) )
#   assert len(predictions) == len(b64_images)


def prepare_synapse(input_data, modality):
    if isinstance(input_data, torch.Tensor):
        input_data = transforms.ToPILImage()(input_data.cpu().detach())
    if isinstance(input_data, list) and isinstance(input_data[0], torch.Tensor):
        for i, img in enumerate(input_data):
            input_data[i] = transforms.ToPILImage()(img.cpu().detach())

    if modality == "image":
        return prepare_image_synapse(input_data)
    elif modality == "video":
        bt.logging.error("Video synapse not implemented yet")
    else:
        raise NotImplementedError(f"Unsupported modality: {modality}")


def prepare_image_synapse(image: Image):
    """
    Prepares an image for use with ImageSynapse object.

    Args:
        image (Image): The input image to be prepared.

    Returns:
        ImageSynapse: An instance of ImageSynapse containing the encoded image and a default prediction value.
    """
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    b64_encoded_image = base64.b64encode(image_bytes.getvalue())
    return ExtendedImageSynapse(image=b64_encoded_image)


class ImageSynapse(bt.Synapse):
    """
    This protocol helps in handling image/prediction request and response communication between
    the miner and the validator.

    Attributes:
    - image: a bas64 encoded images
    - prediction: a float  indicating the probabilty that the image is AI generated/modified.
        >.5 is considered generated/modified, <= 0.5 is considered real.
    """

    testnet_label: int = -1  # for easier miner eval on testnet

    # Required request input, filled by sending dendrite caller.
    image: str = pydantic.Field(title="Image", description="A base64 encoded image", default="", frozen=False)

    # Optional request output, filled by receiving axon.
    prediction: float = pydantic.Field(
        title="Prediction",
        description="Probability that the image is AI generated/modified",
        default=-1.0,
        frozen=False,
    )

    def deserialize(self) -> float:
        """
        Deserialize the output. This method retrieves the response from
        the miner, deserializes it and returns it as the output of the dendrite.query() call.

        Returns:
        - float: The deserialized miner prediction
        prediction probabilities
        """
        return self.prediction

class ExtendedImageSynapse(ImageSynapse):
     model_url: str = ""