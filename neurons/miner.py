import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import bittensor as bt
import torch
import base64
import time
import typing
import io
import numpy as np
from PIL import Image
from torchvision import transforms, models

from bitmind.base.miner import BaseMinerNeuron
from bitmind.protocol import ImageSynapse


class Miner(BaseMinerNeuron):
    """    
    - Receives a base64-encoded image from the validator.
    - Decodes the image.
    - Uses a pre-trained ResNet model to classify the image (0 = Not Construction, 1 = Construction).
    - Sends the classification result back to the validator.
    """

    def __init__(self, config=None, model_name="resnet18"):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Initializing miner for construction site classification.")

        # Load Pre-Trained Model
        self.model = self.load_pretrained_model(model_name)
        self.model.eval()

        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(forward_fn=self.forward_image)

        bt.logging.info(f"Axon created: {self.axon}")

    def load_pretrained_model(self, model_name):
        """
        Loads a pre-trained model from Torchvision.
        """
        model_dict = {
            "resnet18": models.resnet18(pretrained=True),
            "mobilenet_v3": models.mobilenet_v3_large(pretrained=True),
            "efficientnet_b0": models.efficientnet_b0(pretrained=True)
        }
        
        if model_name not in model_dict:
            bt.logging.error(f"Model '{model_name}' not supported! Using ResNet18 as default.")
            model_name = "resnet18"

        model = model_dict[model_name]

        # Modify the last layer for binary classification
        num_features = model.fc.in_features if hasattr(model, "fc") else model.classifier[1].in_features
        model.fc = torch.nn.Linear(num_features, 2)  # 2 output classes: Construction / Non-Construction

        model = model.to(torch.device("cpu"))
        bt.logging.info(f"Loaded pre-trained {model_name} model for classification.")
        return model

    def decode_image(self, image_str):
        image_data = base64.b64decode(image_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    def preprocess_image(self, image):
        """
        Preprocesses the image before passing it to the model.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension

    def classify_image(self, image):
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            logits = self.model(image_tensor)
            prediction = torch.argmax(logits, dim=1).item()  # Get classification result
        return prediction

    async def forward_image(self, synapse: ImageSynapse) -> ImageSynapse:
        """
        Processes the image classification request from the validator.
        - Decodes the base64 image.
        - Classifies it using the pre-trained model.
        - Returns the classification result to the validator.

        Args:
            synapse (ImageSynapse): The synapse object containing the encoded image.

        Returns:
            ImageSynapse: The synapse object with the classification result.
        """
        try:
            bt.logging.info("Received image challenge!")

            # Decode the base64-encoded image
            image = self.decode_image(synapse.image)

            # Classify the image
            synapse.prediction = self.classify_image(image)

            bt.logging.info(f"Prediction = {synapse.prediction}")

        except Exception as e:
            bt.logging.error("Error during image classification")
            bt.logging.error(e)

        return synapse

    def save_state(self):
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    model_name = "resnet18"  # Change this to "mobilenet_v3" or "efficientnet_b0" if needed
    
    with Miner(model_name=model_name) as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(5)