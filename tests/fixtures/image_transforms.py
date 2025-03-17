from functools import partial

import torchvision.transforms as transforms

from natix.utils.image_transforms import (
    ConvertToRGB,
    RandomHorizontalFlipWithParams,
    RandomResizedCropWithParams,
    RandomRotationWithParams,
    RandomVerticalFlipWithParams,
    center_crop,
    get_base_transforms,
    get_random_augmentations,
)
from natix.validator.config import TARGET_IMAGE_SIZE

TRANSFORMS = [
    center_crop,
    RandomHorizontalFlipWithParams,
    RandomVerticalFlipWithParams,
    partial(RandomRotationWithParams, degrees=20, interpolation=transforms.InterpolationMode.BILINEAR),
    partial(RandomResizedCropWithParams, size=TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0)),
    ConvertToRGB,
]

TRANSFORM_PIPELINES = [get_base_transforms(TARGET_IMAGE_SIZE), get_random_augmentations(TARGET_IMAGE_SIZE)]
