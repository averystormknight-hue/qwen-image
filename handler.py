"""
RunPod Serverless Handler for Qwen-Image
"""
import runpod
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)
import torch
from PIL import Image
import base64
import io
import os
import urllib.request
import hashlib
from typing import Optional# Global model instance (loaded once on cold start)
pipeline = None

# LoRA cache directory on network volume
LORA_CACHE_DIR = "/runpod-volume/lora_cache"
os.makedirs(LORA_CACHE_DIR, exist_ok=True)

# Track loaded LoRAs
loaded_loras = {}

def download_lora(url: str) -> str:
    """Download LoRA from URL and cache it. Returns local file path."""
    # Generate filename from URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()
    filename = f"lora_{url_hash}.safetensors"
    filepath = os.path.join(LORA_CACHE_DIR, filename)
    
    # Return cached file if exists
    if os.path.exists(filepath):
        print(f"✅ Using cached LoRA: {filename}")
        return filepath
    
    # Download LoRA
    print(f"📥 Downloading LoRA from: {url}")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ LoRA downloaded and cached: {filename}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to download LoRA: {e}")
        return None

# Available schedulers/samplers
SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "kdpm2": KDPM2DiscreteScheduler,
    "kdpm2_a": KDPM2AncestralDiscreteScheduler,
}

def load_model():
    """Load model once during cold start"""
    global pipeline
    if pipeline is not None:
        return pipeline

    print("🚀 Loading Qwen-Image model...")

    model_name = "Qwen/Qwen-Image"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipeline = pipeline.to(device)

    print(f"✅ Model loaded on {device}")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    return pipeline

def generate_image(job):
    """
    RunPod handler function - mirrors generate_image() from runpod_startup.sh lines 88-112
    Input format: {"input": {"prompt": "...", "width": 1024, ...}}
    Output format: {"image": "base64...", "seed": 123}
    """
    job_input = job["input"]

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    negative_prompt = job_input.get("negative_prompt", " ")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    num_inference_steps = job_input.get("num_inference_steps", 50)
    true_cfg_scale = job_input.get("true_cfg_scale", 4.0)
    seed = job_input.get("seed", None)
    scheduler_name = job_input.get("scheduler", None)  # Optional scheduler/sampler selection
    lora_scale = job_input.get("lora_scale", 1.0)  # LoRA weight scale (0.0 to 1.0)
    lora_url = job_input.get("lora_url", None)  # Optional LoRA URL to download and use

    print(f"🎨 Generating: {prompt[:100]}...")

    # Load model if not already loaded
    pipe = load_model()
    
    # Handle LoRA loading
    if lora_url:
        lora_path = download_lora(lora_url)
        if lora_path:
            # Unload previous LoRA if different
            if lora_url in loaded_loras:
                print("🔄 LoRA already loaded")
            else:
                # Unload all previous LoRAs
                if loaded_loras:
                    print("🔄 Unloading previous LoRA...")
                    pipe.unload_lora_weights()
                    loaded_loras.clear()
                
                # Load new LoRA
                print("🎨 Loading LoRA weights...")
                pipe.load_lora_weights(LORA_CACHE_DIR, weight_name=os.path.basename(lora_path))
                loaded_loras[lora_url] = lora_path
                print("✅ LoRA loaded successfully")
        else:
            print("⚠️ Proceeding without LoRA due to download failure")
    elif lora_scale == 0.0 and loaded_loras:
        # Unload LoRA if scale is 0
        print("🔄 Unloading LoRA (scale=0)...")
        pipe.unload_lora_weights()
        loaded_loras.clear()

    # Set scheduler if specified
    if scheduler_name and scheduler_name.lower() in SCHEDULERS:
        print(f"🔧 Using scheduler: {scheduler_name}")
        scheduler_class = SCHEDULERS[scheduler_name.lower()]
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    elif scheduler_name:
        print(f"⚠️ Unknown scheduler '{scheduler_name}', using default")

    # Setup generator for seed
    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    # Generate image
    with torch.inference_mode():
        # Prepare cross_attention_kwargs for LoRA scale
        cross_attention_kwargs = {"scale": lora_scale} if lora_scale != 1.0 else None
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs
        )

    # Convert to base64
    image = result.images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    used_seed = seed if seed is not None else (generator.initial_seed() if generator else 0)

    print(f"✅ Generated successfully! Seed: {used_seed}")

    return {
        "image": img_b64,
        "seed": used_seed
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": generate_image})
