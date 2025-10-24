#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from datetime import datetime

import warnings
import imageio
import numpy as np
import torch
from diffusers import LTXPipeline
from memory_monitor import MemoryMonitor

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate single LTX-Video from config file')
parser.add_argument('config', type=str, help='Path to JSON config file')
args = parser.parse_args()

# Load configuration from JSON file
if not os.path.exists(args.config):
    print(f"Error: Config file '{args.config}' not found")
    sys.exit(1)

with open(args.config, 'r') as f:
    config = json.load(f)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set memory allocator to reduce fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Load settings from config
MODEL_ID = "Lightricks/LTX-Video"

# Width and height: must be divisible by 32, recommended under 1280×720 (min: 32, max: ~1280)
W, H = config["width"], config["height"]
FPS = 25  # Frames per second (LTX native is 25 FPS)

# Denoising steps: higher = better quality but slower (min: 1, recommended: 20-30 fast / 40+ quality, distilled: 4-10, max: ~200)
STEPS = config["steps"]

# Guidance scale: controls prompt adherence (min: 1.0, recommended: 3.0-3.5, max: ~7.0, distilled models: 1.0)
GUIDE = config["guidance"]

# Guidance rescale: prevents overexposure issues (min: 0.0 = disabled, max: 1.0)
GUIDE_RESCALE = config["guidance_rescale"]

# Decode timestep: improves VAE decoding quality (min: 0.0, recommended: 0.05, max: 1.0)
DECODE_TIMESTEP = config["decode_timestep"]

# Decode noise scale: interpolation between noise and denoised latents at decode (min: 0.0, recommended: 0.025, max: 1.0)
DECODE_NOISE_SCALE = config["decode_noise_scale"]

# Max sequence length: maximum prompt tokens, increase for longer/complex prompts (min: 1, recommended: 128, max: 512)
MAX_SEQ_LEN = config["max_sequence_length"]

# Calculate frame count: LTX requires frames in format (8n + 1), e.g. 97, 105, 113, 257
# Duration in seconds: will be adjusted to nearest valid frame count (max recommended: 257 frames = 10.28s)
DURATION = config["duration"]
desired_frames = int(DURATION * FPS)
# Round to nearest valid frame count (8n + 1)
n = round((desired_frames - 1) / 8)
FRAMES = max(1, 8 * n + 1)  # Ensure at least 1 frame
actual_duration = FRAMES / FPS
SCENE_NAME = config["scene_name"]
OUTDIR = SCENE_NAME
os.makedirs(OUTDIR, exist_ok=True)

# Get prompts from config
prompt = config["prompt"]
neg = config["negative_prompt"]

# Initialize and start memory monitoring (before model loading to capture everything)
monitor = MemoryMonitor(gpu_id=0, sample_interval=0.1)
monitor.start()

# Load LTX-Video model
print(f"Loading LTX-Video model for {actual_duration:.2f}s generation ({FRAMES} frames)...")
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pipe = LTXPipeline.from_pretrained(
    MODEL_ID,
    #dtype=torch.bfloat16 if use_bf16 else torch.float16
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
)
pipe.to("cuda")
pipe.enable_sequential_cpu_offload()

print("Model loaded successfully!")

# Generate video
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
seed = random.randint(1, 2 ** 31 - 1)
output_filename = f"{SCENE_NAME}_{timestamp}_{actual_duration:.2f}sec.mp4"
output_path = f"{OUTDIR}/{output_filename}"

print(f"Generating {FRAMES} frames ({actual_duration:.2f}s) video with seed {seed}...")
g = torch.Generator("cuda").manual_seed(seed)

video = pipe(
    prompt=prompt,
    negative_prompt=neg,
    num_inference_steps=STEPS,
    guidance_scale=GUIDE,
    guidance_rescale=GUIDE_RESCALE,
    num_frames=FRAMES,
    height=H,
    width=W,
    decode_timestep=DECODE_TIMESTEP,
    decode_noise_scale=DECODE_NOISE_SCALE,
    max_sequence_length=MAX_SEQ_LEN,
    generator=g
).frames[0]

# Convert frames - LTX outputs PIL Images already in uint8 format
frames = [np.array(f) for f in video]
imageio.mimwrite(output_path, frames, fps=FPS, quality=9)
print(f"✓ Video written: {output_path}")

# Stop memory monitoring and get peak usage
peak_vram_gb, peak_ram_gb = monitor.stop()
monitor.shutdown()

# Clear GPU memory
torch.cuda.empty_cache()

# Save parameters
params = {
    "scene_name": SCENE_NAME,
    "prompt": prompt,
    "negative_prompt": neg,
    "seed": seed,
    "model_id": MODEL_ID,
    "width": W,
    "height": H,
    "fps": FPS,
    "duration_seconds": actual_duration,
    "duration_requested": DURATION,
    "num_frames": FRAMES,
    "num_inference_steps": STEPS,
    "guidance_scale": GUIDE,
    "guidance_rescale": GUIDE_RESCALE,
    "decode_timestep": DECODE_TIMESTEP,
    "decode_noise_scale": DECODE_NOISE_SCALE,
    "max_sequence_length": MAX_SEQ_LEN,
    "video_quality": 9,
    "timestamp": timestamp,
    "peak_vram_gb": round(peak_vram_gb, 1),
    "peak_ram_gb": round(peak_ram_gb, 1)
}
param_file = os.path.splitext(output_path)[0] + ".json"
with open(param_file, 'w') as f:
    json.dump(params, f, indent=2)
print(f"✓ Parameters written: {param_file}")

print(f"\n✓ Generation complete!")
print(f"  Duration: {actual_duration:.2f}s ({FRAMES} frames)")
print(f"  Resolution: {W}×{H}")
print(f"  Steps: {STEPS}")
print(f"  Guidance: {GUIDE}")
print(f"  Output: {output_path}")
print(f"\n  Memory Usage:")
print(f"    Peak VRAM: {peak_vram_gb:.1f} GB")
print(f"    Peak RAM: {peak_ram_gb:.1f} GB")
