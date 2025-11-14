# Qwen-Image Fast-API

[![Runpod](https://api.runpod.io/badge/pqhaz3925/qwen-image)](https://console.runpod.io/hub/pqhaz3925/qwen-image)
A production-ready RunPod serverless endpoint for Alibaba's Qwen-Image model - a powerful text-to-image generation model with superior text rendering capabilities in both English and Chinese.

## Features

- **Official Qwen-Image Model** - 20B MMDiT image foundation model
- **GPU Optimized** - Runs on A100 80GB, H100 PCIe, H100 HBM3, H100 NVL, and high-end workstation GPUs
- **Auto-scaling** - Scales to 0 when idle to save costs
- **Network Volume Storage** - Model cached persistently across all workers
- **Fast Cold Starts** - Optimized Docker image with pre-installed dependencies

## Model Specifications

- **Model**: `Qwen/Qwen-Image` (20B parameters)
- **Recommended VRAM**: 80GB (A100/H100 recommended)
- **Precision**: bfloat16 (CUDA) / float32 (CPU)
- **Default Resolution**: 1024x1024
- **Text Rendering**: Exceptional quality for both English and Chinese text
- **License**: Apache 2.0

## API Usage

### Input Format

```json
{
  "input": {
    "prompt": "Your image description here",
    "negative_prompt": " ",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": null,
    "scheduler": "euler",
    "lora_url": "https://civitai.com/api/download/models/2316696?type=Model&format=SafeTensor",
    "lora_scale": 1.0
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Description of the image to generate |
| `negative_prompt` | string | `" "` | What to avoid in the image |
| `width` | integer | `1024` | Image width (pixels) |
| `height` | integer | `1024` | Image height (pixels) |
| `num_inference_steps` | integer | `50` | Number of denoising steps (higher = better quality, slower) |
| `true_cfg_scale` | float | `4.0` | Classifier-free guidance scale |
| `seed` | integer | `null` | Random seed for reproducibility (optional) |
| `scheduler` | string | `null` | Sampler/scheduler to use: `euler`, `euler_a`, `dpm`, `ddim`, `pndm`, `lms`, `kdpm2`, `kdpm2_a` |
| `lora_url` | string | `null` | URL to LoRA file (.safetensors) - downloaded and cached automatically |
| `lora_scale` | float | `1.0` | LoRA weight strength (0.0 to 1.0, higher = more LoRA influence) |

### Output Format

```json
{
  "image": "base64_encoded_png_data",
  "seed": 12345
}
```

### Example Request (Python)

```python
import runpod
import base64
from PIL import Image
import io

runpod.api_key = "your_api_key_here"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

request = {
    "input": {
        "prompt": "A serene mountain landscape with Chinese calligraphy 'Harmony'",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "seed": 42,
        "scheduler": "euler",  # Optional: specify sampler
        "lora_url": "https://civitai.com/api/download/models/2316696?type=Model&format=SafeTensor",  # Optional: LoRA URL
        "lora_scale": 1.0      # Optional: adjust LoRA strength
    }
}

run = endpoint.run_sync(request)

# Decode and save image
img_data = base64.b64decode(run['image'])
image = Image.open(io.BytesIO(img_data))
image.save('output.png')

print(f"Generated with seed: {run['seed']}")
```

### Example Request (cURL)

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A futuristic cityscape at sunset",
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 50,
      "scheduler": "dpm",
      "lora_scale": 0.8
    }
  }'
```

## Deployment Configuration

The template is configured with optimal settings in `runpod.toml`:

- **GPU Types**: A100 80GB PCIe, H100 PCIe, H100 HBM3, H100 NVL, RTX 6000 Blackwell, RTX 6000 Blackwell Workstation, RTX Pro 6000 Max-Q Workstation
- **Recommended VRAM**: 80GB
- **Container Disk**: 5GB (code + dependencies)
- **Network Volume**: ~100GB (persistent model storage) - **⚠️ REQUIRED**
- **Workers**: 0-3 (auto-scaling)
- **Timeout**: 600 seconds per job

### ⚠️ Important: Network Volume Required

**You MUST attach a network volume (~100GB) when deploying this endpoint.**

The Qwen-Image model is ~57GB and requires significant disk space. Without a network volume:
- ❌ Deployment will fail due to insufficient disk space
- ❌ Model cannot be downloaded or cached
- ❌ Workers will crash during initialization

The network volume:
- ✅ Stores the model persistently across all workers
- ✅ Prevents re-downloading the model on every cold start
- ✅ Enables faster scaling and startup times

## Performance

- **Cold Start**: ~60-120 seconds (model download on first run)
- **Warm Inference**: ~20-40 seconds (depends on steps and resolution)
- **Memory Usage**: ~50-60GB VRAM for 1024x1024 images (20B parameter model)

## LoRA Support

This endpoint supports **dynamic LoRA loading** - simply provide a `lora_url` in your request to download and use any compatible LoRA:

- **Automatic Download**: LoRAs are downloaded from the provided URL on first use
- **Persistent Caching**: Downloaded LoRAs are cached on the network volume and reused across all workers
- **No Rebuilds Needed**: Change LoRAs without rebuilding the Docker container

### Using LoRAs

```json
{
  "input": {
    "prompt": "your prompt here",
    "lora_url": "https://civitai.com/api/download/models/2316696?type=Model&format=SafeTensor",
    "lora_scale": 1.0
  }
}
```

### LoRA Scale Control

- `lora_scale: 1.0` - Full LoRA strength (default)
- `lora_scale: 0.5` - 50% LoRA influence (blend with base model)
- `lora_scale: 0.0` - Base model only (disables LoRA)

### Supported LoRA Sources

- CivitAI direct download links
- Hugging Face model files
- Any direct URL to a `.safetensors` file

## Scheduler/Sampler Options

Choose different samplers to control the denoising process and image quality:

- **`euler`** - Euler Discrete (fast, good quality)
- **`euler_a`** - Euler Ancestral (more creative, adds randomness)
- **`dpm`** - DPM++ Multistep (high quality, efficient)
- **`ddim`** - DDIM (deterministic, stable)
- **`pndm`** - PNDM (default, good balance)
- **`lms`** - Linear Multistep (smooth results)
- **`kdpm2`** - Karras DPM2 (high quality)
- **`kdpm2_a`** - Karras DPM2 Ancestral (creative variant)

Different schedulers can significantly affect generation speed and image characteristics. Experiment to find the best one for your use case!

## Tips for Best Results

1. **Prompt Quality**: Be specific and descriptive
2. **Steps**: 30-50 steps for good quality, 50-100 for best quality
3. **CFG Scale**: 3.5-5.0 works well for most prompts
4. **Text Rendering**: Qwen-Image excels at rendering text - great for logos, signs, and calligraphy
5. **Seed**: Use the same seed to reproduce images
6. **Scheduler Selection**: Try `euler` or `dpm` for faster generation, `euler_a` for more creative results
7. **LoRA Usage**: Include `lora_url` to use any compatible LoRA (cached automatically)
8. **LoRA Tuning**: Reduce `lora_scale` (0.5-0.7) if the LoRA effect is too strong

## License

This endpoint uses the Qwen-Image model licensed under Apache 2.0. For more information, visit the [official Qwen-Image repository](https://github.com/QwenLM/Qwen-Image).

## Support

For issues or questions about this RunPod template, please open an issue on the GitHub repository.
