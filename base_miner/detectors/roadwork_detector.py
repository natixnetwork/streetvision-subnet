from PIL import Image

from base_miner.detectors import FeatureDetector
from base_miner.gating_mechanisms import GatingMechanism
from base_miner.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register_module(module_name="ROADWORK")
class RoadworkDetector(FeatureDetector):
    """
    This DeepfakeDetector subclass implements Content-Aware Model Orchestration
    (CAMO), a mixture-of-experts approach to the binary classification of
    real and fake images, breaking the classification problem into content-specific
    subproblems.

    The subproblems are solved by using a GatingMechanism to route image
    content to appropriate DeepfakeDetector subclass instance(s) that
    initialize models pretrained to handle the content type.

    Attributes:
        model_name (str): Name of the detector instance.
        config_name (str): Name of the YAML file in deepfake_detectors/config/ to load
                      attributes from.
        device (str): The type of device ('cpu' or 'cuda').
    """

    def __init__(self, model_name: str = "ROADWORK", config_name: str = "roadwork.yaml", device: str = "cpu"):
        """
        Initialize the Detector with dynamic model selection based on config.
        """
        self.detectors = {}
        super().__init__(model_name, config_name, device)

        gate_names = [
            content_type for content_type in self.content_type if self.content_type[content_type].get("use_gate", False)
        ]
        self.gating_mechanism = GatingMechanism(gate_names)

    def load_model(self):
        """
        Load detectors dynamically based on the provided configuration and registry.
        """
        for content_type, detector_info in self.content_type.items():
            model_name = detector_info["model"]
            detector_config = detector_info["detector_config"]
            if model_name in DETECTOR_REGISTRY:
                self.detectors[content_type] = DETECTOR_REGISTRY[model_name](
                    model_name=f"{model_name}_{content_type}", config_name=detector_config, device=self.device
                )
            else:
                raise ValueError(f"Detector {model_name} not found in the registry for {content_type}.")

    def __call__(self, image: Image) -> float:
        """
        Perform inference using the CAMO detector.

        Args:
            image (PIL.Image): The input image to classify.

        Returns:
            float: The prediction score indicating the likelihood of the image being a deepfake.
        """
        # gate_results = self.gating_mechanism(image)
        # bt.logging.debug(f"Gate results: {gate_results}")
        # expert_outputs = {}
        # for content_type, gate_output_image in gate_results.items():
        #     pred = self.detectors[content_type](gate_output_image)
        #     bt.logging.debug(f"Detector {content_type} output: {pred}")
        #     expert_outputs[content_type] = pred

        # if len(expert_outputs) == 0:
        #     return self.detectors['general'](image)
        pred = self.detectors["roadwork"](image)
        return pred
