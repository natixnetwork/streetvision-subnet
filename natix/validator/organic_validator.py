from typing import Dict, List, Optional, Tuple

import bittensor as bt
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class OrganicValidator:
    """
    Validator for organic requests using Florence-2 vision model.
    Provides object detection, captioning, and explanation verification.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.torch_dtype = torch_dtype if self.device == "cuda" else torch.float32

        bt.logging.info(f"Loading Florence-2 model: {model_name} on {self.device}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            bt.logging.success(f"Florence-2 model loaded successfully on {self.device}")

        except Exception as e:
            bt.logging.error(f"Failed to load Florence-2 model: {e}")
            raise

    def _run_inference(self, image: Image.Image, task_prompt: str) -> str:
        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def detect_objects(self, image: Image.Image) -> Dict[str, List]:
        """
        Detect objects in the image.

        Returns:
            Dict with 'labels' and 'bboxes' keys
        """
        try:
            result = self._run_inference(image, '<OD>')
            od_result = result.get('<OD>', {'labels': [], 'bboxes': []})

            bt.logging.debug(f"Florence-2 detected objects: {od_result['labels']}")
            return od_result

        except Exception as e:
            bt.logging.error(f"Object detection failed: {e}")
            return {'labels': [], 'bboxes': []}

    def generate_caption(self, image: Image.Image, detailed: bool = True) -> str:
        try:
            task = '<DETAILED_CAPTION>' if detailed else '<CAPTION>'
            result = self._run_inference(image, task)

            caption = result.get(task, "")
            bt.logging.debug(f"Florence-2 caption: {caption}")
            return caption

        except Exception as e:
            bt.logging.error(f"Caption generation failed: {e}")
            return ""

    def predict_roadwork(self, image: Image.Image) -> Tuple[float, str]:
        """
        Predict if image contains roadwork.

        Returns:
            Tuple of (prediction_score 0-1, reasoning string)
        """
        objects = self.detect_objects(image)
        caption = self.generate_caption(image, detailed=True)

        detected_labels = [label.lower() for label in objects['labels']]
        caption_lower = caption.lower()

        roadwork_indicators = [
            'traffic cone', 'cone', 'construction cone',
            'barrier', 'construction barrier', 'road barrier',
            'excavator', 'bulldozer', 'construction equipment',
            'construction sign', 'road work', 'roadwork',
            'caution tape', 'safety vest', 'hard hat',
            'construction vehicle', 'dump truck', 'cement mixer',
            'construction site', 'road construction'
        ]

        object_matches = []
        for indicator in roadwork_indicators:
            for label in detected_labels:
                if indicator in label or label in indicator:
                    object_matches.append(label)
                    break

        caption_matches = []
        for indicator in roadwork_indicators:
            if indicator in caption_lower:
                caption_matches.append(indicator)

        object_score = min(len(object_matches) * 0.3, 0.7)
        caption_score = min(len(caption_matches) * 0.15, 0.3)
        total_score = min(object_score + caption_score, 1.0)

        reasoning_parts = []
        if object_matches:
            reasoning_parts.append(f"Detected: {', '.join(object_matches)}")
        if caption_matches:
            reasoning_parts.append(f"Caption mentions: {', '.join(caption_matches)}")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No roadwork indicators detected"

        bt.logging.debug(f"Florence-2 roadwork prediction: {total_score:.2f} - {reasoning}")

        return total_score, reasoning

    def verify_explanation(
        self,
        image: Image.Image,
        miner_explanation: str,
        miner_prediction: float
    ) -> Dict:
        """
        Verify miner's explanation against Florence-2 analysis.

        Returns:
            Dict with verification metrics
        """
        if not miner_explanation or miner_explanation.strip() == "":
            return {
                'objects_verified': 0.0,
                'false_claims': [],
                'specificity_score': 0.0,
                'florence_prediction': 0.0,
                'florence_reasoning': "No explanation provided",
                'semantic_match': 0.0
            }

        florence_pred, florence_reasoning = self.predict_roadwork(image)
        objects = self.detect_objects(image)
        caption = self.generate_caption(image, detailed=True)

        detected_labels = [label.lower() for label in objects['labels']]
        miner_explanation_lower = miner_explanation.lower()

        roadwork_keywords = [
            'cone', 'barrier', 'excavator', 'bulldozer', 'sign',
            'equipment', 'vehicle', 'truck', 'vest', 'hat',
            'construction', 'roadwork', 'caution', 'tape'
        ]

        claimed_objects = [kw for kw in roadwork_keywords if kw in miner_explanation_lower]

        verified_count = 0
        false_claims = []

        for claimed in claimed_objects:
            found = False
            for detected in detected_labels:
                if claimed in detected or detected in claimed:
                    found = True
                    break

            if found:
                verified_count += 1
            else:
                if claimed not in caption.lower():
                    false_claims.append(claimed)

        objects_verified = verified_count / len(claimed_objects) if claimed_objects else 0.0

        word_count = len(miner_explanation.split())
        specificity_score = min(word_count / 20.0, 1.0)

        miner_words = set(miner_explanation_lower.split())
        caption_words = set(caption.lower().split())
        common_words = miner_words & caption_words
        semantic_match = len(common_words) / len(miner_words) if miner_words else 0.0

        return {
            'objects_verified': objects_verified,
            'false_claims': false_claims,
            'specificity_score': specificity_score,
            'florence_prediction': florence_pred,
            'florence_reasoning': florence_reasoning,
            'semantic_match': semantic_match
        }

    def free_memory(self):
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model

        if hasattr(self, 'processor'):
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bt.logging.info("Florence-2 model memory freed")
