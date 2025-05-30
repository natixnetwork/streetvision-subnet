import gc
import os
import random
import warnings

import bittensor as bt
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

from base_miner.detectors import FeatureDetector
from base_miner.registry import DETECTOR_REGISTRY

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Ignore INFO and WARN messages
warnings.filterwarnings("ignore", category=FutureWarning)


@DETECTOR_REGISTRY.register_module(module_name="ViT")
class ViTImageDetector(FeatureDetector):
    """
    ViTImageDetector subclass that initializes a pretrained model
    for binary classification of roadwork.

    Attributes:
        model_name (str): Name of the detector instance.
        config_name (str): Name of the YAML file in detectors/config/ to load
                      attributes from.
        device (str): The type of device ('cpu' or 'cuda').
    """

    def __init__(self, model_name: str = "ViT", config_name: str = "ViT_roadwork.yaml", device: str = "cpu"):
        super().__init__(model_name, config_name, device)

    def init_seed(self):
        seed_value = self.config.get("manualSeed")
        if seed_value:
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)

    def load_model(self):
        self.model = pipeline(
            "image-classification",
            model=AutoModelForImageClassification.from_pretrained(self.hf_repo),
            feature_extractor=AutoImageProcessor.from_pretrained(self.hf_repo, use_fast=True),
        )

    def preprocess(self, image, res=256):
        """Preprocess the image for model inference.

        Returns:
            torch.Tensor: The preprocessed image tensor, ready for model inference.
        """
        # Convert image to RGB format to ensure consistent color handling.
        image = image.convert("RGB")
        if "shortest_edge" in self.model.feature_extractor.size:
            size = self.model.feature_extractor.size["shortest_edge"]
        else:
            (self.model.feature_extractor.size["height"], self.model.feature_extractor.size["width"])
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.model.feature_extractor.image_mean, std=self.model.feature_extractor.image_std),
            ]
        )

        # Apply transformations and add a batch dimension for model inference.
        image_tensor = transform(image).unsqueeze(0)

        # Move the image tensor to the specified device (e.g., GPU).
        return image_tensor.to(self.device)

    def infer(self, image_tensor):
        """Perform inference using the model."""
        with torch.no_grad():
            self.model({"image": image_tensor}, inference=True)
        return self.model.prob[-1]

    def __call__(self, image: Image) -> float:
        # image_tensor = self.preprocess(image)
        # output = self.infer(image_tensor)
        bt.logging.debug(f"{image}")
        output = self.model(image)  # pipeline handles preprocessing
        # result eg. [{'label': 'Roadwork', 'score': 0.9815}, {'label': 'None', 'score': 0.0184}]
        output = self.convert_output(output)
        bt.logging.debug(f"Model output: {output}")
        return output["Roadwork"]

    def convert_output(self, result):
        new_output = {}
        for item in result:
            new_output[item["label"]] = item["score"]
        return new_output

    def free_memory(self):
        """Frees up memory by setting model and large data structures to None."""
        if self.model is not None:
            self.model.cpu()  # Move model to CPU to free up GPU memory (if applicable)
            del self.model
            self.model = None

        if self.face_detector is not None:
            del self.face_detector
            self.face_detector = None

        if self.face_predictor is not None:
            del self.face_predictor
            self.face_predictor = None

        gc.collect()

        # If using GPUs and PyTorch, clear the cache as well
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
