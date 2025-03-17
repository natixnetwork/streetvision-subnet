import numpy as np
from PIL import Image

from base_miner.gating_mechanisms import Gate
from base_miner.registry import GATE_REGISTRY


@GATE_REGISTRY.register_module(module_name="ROADWORK")
class RoadworkGate(Gate):
    """
    Gate subclass for roadwork content detection and preprocessing.

    Attributes:
        gate_name (str): The name of the gate.
        predictor_path (str): Path to dlib face landmark model.
    """

    def __init__(self, gate_name: str = "RoadworkGate"):
        super().__init__(gate_name, "roadwork")

    def preprocess(self, image: np.ndarray, res=256) -> any:
        """
        Align and crop the largest face in the image

        Args:
            image: Input image array
            faces: Output out of a dlib face detection model
            res: NxN image size

        Returns:
            preprocessed image with largest face aligned and cropped
        """

        return image

    def __call__(self, image: Image, res: int = 256) -> any:
        """
        Perform face detection and image aligning and cropping to the face.

        Args:
            image (PIL.Image): The image to classify and preprocess if content is detected.

        Returns:
            image (PIL.Image): The processed face image or original image if no faces.
        """
        image_np = np.array(image)
        faces = self.face_detector(image_np, 1)
        if faces is None or len(faces) == 0:
            return image, False

        processed_image = self.preprocess(image_np, faces, res)
        return processed_image, True
