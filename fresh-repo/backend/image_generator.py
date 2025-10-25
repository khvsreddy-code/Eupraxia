"""
Multimodal support for image generation using Stable Diffusion.
Optimized for 8GB RAM systems using CPU.
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = "cpu"
        self.setup_pipeline()

    def setup_pipeline(self):
        logger.info("Setting up Stable Diffusion pipeline with optimizations...")
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,  # Use float32 for CPU
                safety_checker=None,  # Disable for memory savings
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            self.pipe.enable_attention_slicing(slice_size="auto")
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_vae_slicing()
            
            # Enable model CPU offload
            self.pipe.enable_model_cpu_offload()
            
            logger.info("Pipeline setup complete with all optimizations enabled")
        except Exception as e:
            logger.error(f"Error setting up pipeline: {e}")
            raise

    def generate_image(self, prompt, output_path="generated_image.png", 
                      num_inference_steps=20, guidance_scale=7.5):
        """
        Generate image from text prompt
        
        Args:
            prompt (str): Text description of desired image
            output_path (str): Where to save the generated image
            num_inference_steps (int): Number of denoising steps (lower = faster)
            guidance_scale (float): How closely to follow the prompt
        """
        logger.info(f"Generating image for prompt: {prompt}")
        try:
            # Generate image
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            # Save image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            logger.info(f"Image saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    generator = ImageGenerator()
    output = generator.generate_image(
        "A serene landscape with mountains and a lake at sunset",
        "outputs/landscape.png"
    )
    print(f"Generated image saved to: {output}")