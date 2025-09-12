import gc

import bittensor as bt
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor
from transformers import logging as transformers_logging
from transformers import pipeline

from natix.validator.config import HUGGINGFACE_CACHE_DIR



class PromptGenerator:
    """
    A class for generating and moderating image annotations using transformer models.

    This class provides functionality to generate descriptive captions for images
    using BLIP2 models and optionally moderate the generated text using a separate
    language model.
    """

    def __init__(
        self,
        vlm_name: str,
        llm_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> None:
        """
        Initialize the ImageAnnotationGenerator with specific models and device settings.

        Args:
            model_name: The name of the BLIP model for generating image captions.
            text_moderation_model_name: The name of the model used for moderating
                text descriptions.
            device: The device to use.
            apply_moderation: Flag to determine whether text moderation should be
                applied to captions.
        """
        self.vlm_name = vlm_name
        self.llm_name = llm_name
        self.vlm_processor = None
        self.vlm = None
        self.llm_pipeline = None
        self.device = device

    def are_models_loaded(self) -> bool:
        return (self.vlm is not None) and (self.llm_pipeline is not None)

    def load_models(self) -> None:
        """
        Load the necessary models for image annotation and text moderation onto
        the specified device.
        """
        if self.are_models_loaded():
            bt.logging.warning("Models already loaded")
            return

        bt.logging.info(f"Loading caption generation model {self.vlm_name}")
        self.vlm_processor = Blip2Processor.from_pretrained(self.vlm_name, cache_dir=HUGGINGFACE_CACHE_DIR)
        self.vlm = Blip2ForConditionalGeneration.from_pretrained(
            self.vlm_name, 
            torch_dtype=torch.float32,
            cache_dir=HUGGINGFACE_CACHE_DIR
        )
        self.vlm.to(self.device)
        
        # Convert all float32 parameters to float16 to save memory
        for param in self.vlm.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.float16)
        
        # Enable CPU offloading for memory efficiency
        if hasattr(self.vlm, 'enable_model_cpu_offload'):
            self.vlm.enable_model_cpu_offload()
        bt.logging.info(f"Loaded image annotation model {self.vlm_name}")

        bt.logging.info(f"Loading caption moderation model {self.llm_name}")
        llm = AutoModelForCausalLM.from_pretrained(self.llm_name, torch_dtype=torch.bfloat16, cache_dir=HUGGINGFACE_CACHE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=HUGGINGFACE_CACHE_DIR)
        llm = llm.to(self.device)
        
        # Convert any float32 parameters to float16 for memory efficiency
        for param in llm.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.float16)
                
        self.llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer)
        bt.logging.info(f"Loaded caption moderation model {self.llm_name}")

    def clear_gpu(self) -> None:
        """
        Clear GPU memory by moving models back to CPU and deleting them,
        followed by collecting garbage.
        """
        bt.logging.info("Clearing GPU memory after prompt generation")
        if self.vlm:
            self.vlm.to("cpu")
            del self.vlm
            self.vlm = None

        if self.vlm_processor:
            del self.vlm_processor
            self.vlm_processor = None

        if self.llm_pipeline:
            self.llm_pipeline.model.to("cpu")
            del self.llm_pipeline
            self.llm_pipeline = None

        # Multiple rounds of garbage collection and cache clearing
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        bt.logging.info("GPU memory cleared")

    def generate(self, image: Image.Image, label: int = None, max_new_tokens: int = 80, verbose: bool = False) -> str:
        """
        Generate a string description for a given image using BLIP2 with no prompt.

        Args:
            image: The image for which the description is to be generated.
            label: 0 for no roadwork, 1 for roadwork present, None for generic.
            max_new_tokens: The maximum number of tokens to generate.
            verbose: If True, additional logging information is printed.

        Returns:
            A generated description of the image.
        """
        if not verbose:
            transformers_logging.set_verbosity_error()

        inputs = self.vlm_processor(image, return_tensors="pt").to(self.device)
        
        generated_ids = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if verbose:
            bt.logging.info(f"Generated caption: {caption}")

        if not verbose:
            transformers_logging.set_verbosity_info()

        if caption and not caption.endswith("."):
            caption += "."

        if not caption:
            caption = "Dashcam view of road scene."

        moderated_description = self.moderate(caption, label)
        return moderated_description

    def moderate(self, description: str, label: int = None, max_new_tokens: int = 80) -> str:
        """
        Use the text moderation pipeline to make the description more concise
        and tailored to the specific label.

        Args:
            description: The text description to be moderated.
            label: 0 for no roadwork, 1 for roadwork present, None for generic.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            The moderated description text, or the original description if
            moderation fails.
        """
        if label == 1:
            system_content = (
                "[INST]Enhance this dashcam footage description to include active roadwork. "
                "Start with 'Photorealistic dashcam footage' and ADD roadwork elements to the existing scene: "
                "orange traffic cones, construction barriers, construction vehicles, road crews in safety vests, "
                "lane closure signs, construction equipment, or construction zones. The scene MUST show active roadwork.[/INST]"
            )
        elif label == 0:
            system_content = (
                "[INST]Rewrite as concise dashcam footage description. "
                "Focus on clear roads and regular traffic. "
                "Start with 'Photorealistic dashcam footage' of normal road and keep factual.[/INST]"
            )
        else:
            system_content = (
                "[INST]Rewrite as concise dashcam footage description. "
                "Start with 'Photorealistic dashcam footage' and keep factual.[/INST]"
            )
            
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": description},
        ]
        try:
            moderated_text = self.llm_pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
                return_full_text=False,
            )
            return moderated_text[0]["generated_text"]

        except Exception as e:
            bt.logging.error(f"An error occurred during moderation: {e}", exc_info=True)
            return description
