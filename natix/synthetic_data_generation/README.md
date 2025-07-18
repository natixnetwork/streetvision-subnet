
# Synthetic Image Generation

This folder contains files for generating realistic synthetic images for the StreetVision subnet (Subnet 72).

## How it works

1. **Image Sampling** - Takes real images from the cache with 50/50 roadwork/non-roadwork split
2. **Prompt Generation** - Uses BLIP-2 to create captions, enhanced with photorealistic keywords
3. **Image Generation** - Uses Stable Diffusion models with negative prompts to ensure realism
4. **Storage** - Saves generated images to `~/.cache/natix/Synthetic/`

## Models Used

- **BLIP-2** - Image captioning
- **Llama 3.1** - Text moderation
- **Stable Diffusion XL/2.1** - Image generation

## Key Features

- 50/50 roadwork distribution
- Photorealistic output with negative prompts
- Automatic caching and metadata storage
- Support for text-to-image and image-to-image generation
