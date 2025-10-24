import json
import os
import random
from datetime import datetime

import warnings
import imageio
import numpy as np
import torch
from diffusers import LTXPipeline, LTXImageToVideoPipeline

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set memory allocator to reduce fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Memory optimizations optimized for: RTX 5060 Ti (16GB VRAM), 8-core Xeon CPU, 64GB RAM
# LTX-Video is much more memory efficient than ModelScope - applied after pipe initialization

MODEL_ID = "Lightricks/LTX-Video"
W, H = 640, 480  # Video resolution (keeping at 640x480)
FPS = 25  # Frames per second (LTX native is 25 FPS)
LENGTH = 4  # Video length in seconds (limited by VAE decode VRAM)
FRAMES = 97  # Frame count (~4 seconds at 25fps - fits in VAE decode limit)
STEPS = 40  # Inference steps (maxed out for sharpest quality possible)
GUIDE = 6.0  # Guidance scale (6.0 for strong prompt adherence)
OUTDIR = "clips"
os.makedirs(OUTDIR, exist_ok=True)

prompt = (
    "Wide shot of three people walking from far left edge to far right edge. "
    "People enter from left side of frame, walk across entire frame, exit right side. "
    "Dirt path through philippine village. "
    "Well-lit people with visible faces and clothing details. "
    "Bright natural lighting on people. "
    "Clear human features, realistic skin tones. "
    "Colorful traditional filipino clothing. "
    "Village buildings in background. "
    "Blue sky. "
    "Sharp focus, photorealistic"
)
neg = (
    "Dark silhouettes, shadowy people, black figures, underexposed people. "
    "Faceless, no facial features. "
    "Transparent, ghost-like, see-through, translucent. "
    "Static people, people not moving. "
    "Blurry, hazy, out of focus. "
    "Cropped bodies, people cut off at edges. "
    "Deformed, watermark, low quality"
)

# Load LTX-Video model - much faster than ModelScope
print("Loading LTX-Video model...")
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pipe = LTXPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
)
pipe.to("cuda")

# Memory optimizations - Aggressive CPU offload to maximize quality
# Sequential offload: moves each model component to CPU after use (saves more VRAM, slower)
# Trade-off: ~2-3x slower but allows higher resolution/steps/frames
pipe.enable_sequential_cpu_offload()  # More aggressive than model_cpu_offload

print("Model loaded successfully!")

def gen(prompt, seed, name):
    print(f"Generating video with seed {seed}...")
    g = torch.Generator("cuda").manual_seed(seed)

    video = pipe(
        prompt=prompt,
        negative_prompt=neg,
        num_inference_steps=STEPS,
        guidance_scale=GUIDE,
        num_frames=FRAMES,
        height=H,
        width=W,
        generator=g
    ).frames[0]

    # Convert frames - LTX outputs PIL Images already in uint8 format (0-255)
    # Don't multiply by 255 like ModelScope (which outputs float32 0-1 range)
    frames = [np.array(f) for f in video]
    imageio.mimwrite(name, frames, fps=FPS, quality=9)
    print("wrote", name)

    # Clear GPU memory after generation
    torch.cuda.empty_cache()

    # Save parameters with same name but .json extension
    params = {
        "prompt": prompt,
        "negative_prompt": neg,
        "seed": seed,
        "model_id": MODEL_ID,
        "width": W,
        "height": H,
        "fps": FPS,
        "length_seconds": LENGTH,
        "num_frames": FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDE,
        "video_quality": 9
    }
    param_file = os.path.splitext(name)[0] + ".json"
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=2)
    print("wrote", param_file)


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
seed0 = random.randint(1, 2 ** 31 - 1)
gen(prompt, seed0, f"{OUTDIR}/ltx_{timestamp}.mp4")

# Concatenate multiple clips with:
#   printf "file 'clips/01_lr.mp4'\nfile 'clips/02_idle.mp4'\nfile 'clips/03_rl.mp4'\n" > list.txt
#   ffmpeg -y -f concat -safe 0 -i list.txt -c copy out.mp4
