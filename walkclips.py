#!/usr/bin/env python3
import json
import os
import random
from datetime import datetime

import warnings
import imageio
import numpy as np
import torch
from diffusers import DiffusionPipeline

# Suppress deprecation warnings for TextToVideoSDPipeline
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

# Set memory allocator to reduce fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Memory optimizations optimized for: RTX 5060 Ti (16GB VRAM), 8-core Xeon CPU, 64GB RAM
# Note: CPU offload disabled - old Xeon is slower than keeping everything on GPU
# Applied after pipe initialization below

MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
W, H = 576, 384  # Video resolution (width x height in pixels)
FPS = 8  # Frames per second (playback speed)
LENGTH = 3  # Video length in seconds (reduced from 5 to fit in 16GB VRAM)
FRAMES = FPS * LENGTH  # Computed: total frames to generate (24 frames)
STEPS = 24  # Inference steps (denoising iterations; more = better quality but slower)
GUIDE = 9.0  # Guidance scale (7-8: creative/loose, 9-12: balanced, 13+: strict prompt adherence)
OUTDIR = "clips"
os.makedirs(OUTDIR, exist_ok=True)

prompt = (
    "A static wide shot of a philippine village dirt path in late afternoon golden hour light. "
    "Clear blue sky and horizon line visible in upper third of frame. "
    "Three filipino people in traditional clothing walking naturally from left to right across the frame. "
    "Complete full body visible from head to feet, entire body in frame, people occupy one-third of frame height. "
    "Eye-level camera perspective, open landscape with sky, natural gait and movement, warm natural lighting with soft shadows. "
    "Wide framing with space around people, photorealistic, cinematic, documentary style"
)
neg = (
    "lowres, bad quality, blurry, deformed anatomy, extra limbs, distorted faces, "
    "temporal flickering, smearing, watermark, text overlay, logo, "
    "closeup, cropped bodies, partial people, cut off limbs, missing feet, missing head, cut off at edges, "
    "zoomed in, tight framing, extreme angles, overhead view, low angle, ground level, "
    "artificial lighting, unnatural movement, floating people"
)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Apply memory optimizations
pipe.enable_vae_slicing()  # Process VAE in slices (~5-10% slower, saves ~1-2GB VRAM)
pipe.enable_attention_slicing()  # Slice attention computation (~10-15% slower, saves ~2-3GB VRAM)
pipe.enable_vae_tiling()  # Process frames in tiles (~15-20% slower, saves ~3-4GB VRAM)
# pipe.enable_model_cpu_offload()  # DISABLED: Would save ~4-6GB VRAM but 2-5x slower on old Xeon

def gen(prompt, seed, name):
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=neg,
        num_inference_steps=STEPS,
        guidance_scale=GUIDE,
        num_frames=FRAMES,
        height=H,
        width=W,
        generator=g
    )
    # Convert frames to uint8 to avoid lossy conversion warnings
    frames = [(np.asarray(f) * 255).astype(np.uint8) for f in out.frames[0]]
    imageio.mimwrite(name, frames, fps=FPS, quality=8)
    print("wrote", name)

    # Clear GPU memory after generation
    del out
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
        "video_quality": 8
    }
    param_file = os.path.splitext(name)[0] + ".json"
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=2)
    print("wrote", param_file)


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
seed0 = random.randint(1, 2 ** 31 - 1)
gen(prompt, seed0, f"{OUTDIR}/{timestamp}.mp4")

# Concatenate multiple clips with:
#   printf "file 'clips/01_lr.mp4'\nfile 'clips/02_idle.mp4'\nfile 'clips/03_rl.mp4'\n" > list.txt
#   ffmpeg -y -f concat -safe 0 -i list.txt -c copy out.mp4
