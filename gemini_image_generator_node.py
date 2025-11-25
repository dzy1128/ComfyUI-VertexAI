from google import genai
from google.genai import types
import os
import numpy as np
import torch
from PIL import Image
import io
from datetime import datetime

class GeminiImageGenerator:
    """
    A ComfyUI node for generating images using Google Gemini API
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate an image"
                }),
                "model_name": (["gemini-3-pro-image-preview"], {
                    "default": "gemini-3-pro-image-preview"
                }),
                "image_resolution": (["1K", "2K", "4K", "1280x1920", "1920x1280", "1024x1024"], {
                    "default": "1K"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info_text")
    FUNCTION = "generate_image"
    CATEGORY = "Gemini"
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # ComfyUI uses [B, H, W, C] format with values in [0, 1]
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first batch
        
        # Convert to numpy and scale to 0-255
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Ensure 3 channels (RGB)
        if len(np_image.shape) == 2:  # Grayscale
            np_image = np.stack([np_image] * 3, axis=-1)
        elif np_image.shape[2] == 4:  # RGBA
            np_image = np_image[:, :, :3]
        
        # Add batch dimension [B, H, W, C]
        tensor = torch.from_numpy(np_image)[None,]
        return tensor
    
    def parse_resolution(self, resolution_str):
        """Parse resolution string to get aspect_ratio and image_size"""
        # Handle specific dimensions like "1280x1920"
        if "x" in resolution_str:
            width, height = map(int, resolution_str.split("x"))
            if width == height:
                aspect_ratio = "1:1"
            elif width > height:
                aspect_ratio = "16:9"
            else:
                aspect_ratio = "9:16"
            
            # Convert to closest K size
            max_dim = max(width, height)
            if max_dim <= 1024:
                image_size = "1K"
            elif max_dim <= 2048:
                image_size = "2K"
            else:
                image_size = "4K"
            
            return aspect_ratio, image_size
        else:
            # Handle K sizes
            return "1:1", resolution_str
    
    def generate_image(self, user_prompt, model_name, image_resolution, seed,
                       system_prompt="",
                       image_1=None, image_2=None, image_3=None, 
                       image_4=None, image_5=None, image_6=None):
        
        start_time = datetime.now()
        info_dict = {
            "user_prompt": user_prompt,
            "model_name": model_name,
            "image_resolution": image_resolution,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": False,
            "error": None
        }
        
        try:
            # Initialize Gemini client
            client = genai.Client(
                vertexai=True,
                api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
            )
            
            # Prepare content parts
            content_parts = []
            
            # Add input images if provided
            for idx, img_tensor in enumerate([image_1, image_2, image_3, image_4, image_5, image_6], 1):
                if img_tensor is not None:
                    # Convert tensor to PIL Image
                    pil_image = self.tensor_to_pil(img_tensor)
                    
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Add to content parts
                    content_parts.append(types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png"
                    ))
            
            # Build prompt text
            prompt_text = ""
            if system_prompt:
                prompt_text += f"System: {system_prompt}\n\n"
            prompt_text += user_prompt
            
            # Add text prompt
            content_parts.append(types.Part.from_text(text=prompt_text))
            
            # Create content
            contents = [
                types.Content(
                    role="user",
                    parts=content_parts
                )
            ]
            
            # Parse resolution
            aspect_ratio, image_size = self.parse_resolution(image_resolution)
            
            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                max_output_tokens=32768,
                response_modalities=["TEXT", "IMAGE"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )
                ],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    output_mime_type="image/png",
                ),
            )
            
            # Generate image
            generated_image_data = None
            response_text = []
            
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.candidates:
                    for candidate in chunk.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                # Collect text
                                if hasattr(part, 'text') and part.text:
                                    response_text.append(part.text)
                                
                                # Get image data
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    generated_image_data = part.inline_data.data
            
            # Process generated image
            if generated_image_data:
                pil_image = Image.open(io.BytesIO(generated_image_data))
                output_tensor = self.pil_to_tensor(pil_image)
                
                info_dict["success"] = True
                end_time = datetime.now()
                info_dict["generation_time"] = f"{(end_time - start_time).total_seconds():.2f} seconds"
                
                # Format info text
                info_text = f"""=== Generation Info ===
User Prompt: {user_prompt}
Model Name: {model_name}
Image Resolution: {image_resolution} (aspect_ratio: {aspect_ratio}, size: {image_size})
Start Time: {info_dict["start_time"]}
Generation Time: {info_dict["generation_time"]}
Status: ✅ Success
Seed: {seed}
"""
                if response_text:
                    info_text += f"\nResponse Text:\n{''.join(response_text)}"
                
                return (output_tensor, info_text)
            else:
                raise Exception("No image generated in response")
        
        except Exception as e:
            info_dict["success"] = False
            info_dict["error"] = str(e)
            
            # Format error info text
            info_text = f"""=== Generation Info ===
User Prompt: {user_prompt}
Model Name: {model_name}
Image Resolution: {image_resolution}
Start Time: {info_dict["start_time"]}
Status: ❌ Failed
Error: {str(e)}
Seed: {seed}
"""
            
            # Return a black image as placeholder
            black_image = torch.zeros((1, 512, 512, 3))
            return (black_image, info_text)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerator": GeminiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerator": "Gemini Image Generator"
}

