#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime

import warnings
import imageio
import numpy as np
import torch
from PIL import Image
from diffusers import LTXPipeline, LTXImageToVideoPipeline

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate LTX-Video from config file')
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

# Memory optimizations optimized for: RTX 5060 Ti (16GB VRAM), 8-core Xeon CPU, 64GB RAM

MODEL_ID = "Lightricks/LTX-Video"
W, H = config.get("width", 640), config.get("height", 480)
FPS = 25  # Frames per second (LTX native is 25 FPS)
STEPS = config.get("steps", 40)
GUIDE = config.get("guidance", 6.0)
CLIP_LENGTH = config.get("clip_length", 4)
SCENE_NAME = config["scene_name"]
OUTDIR = SCENE_NAME
os.makedirs(OUTDIR, exist_ok=True)

# Get prompts from config
prompt = config["prompt"]
neg = config["negative_prompt"]

# Calculate clip configurations based on duration and clip length
def calculate_clips(total_duration, clip_length):
    clips = []
    remaining = total_duration
    while remaining > 0:
        length = min(clip_length, remaining)
        frames = int(length * FPS)
        clips.append({"frames": frames, "length": length})
        remaining -= length
    return clips

CLIP_CONFIGS = calculate_clips(config.get("duration", 10), CLIP_LENGTH)

# Load LTX-Video models - text-to-video and image-to-video
print("Loading LTX-Video models...")
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

# Text-to-video pipeline for first clip
pipe_t2v = LTXPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
pipe_t2v.to("cuda")
pipe_t2v.enable_sequential_cpu_offload()

# Image-to-video pipeline for continuation clips
pipe_i2v = LTXImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
pipe_i2v.to("cuda")
pipe_i2v.enable_sequential_cpu_offload()

print("Models loaded successfully!")

def gen_text2video(prompt, seed, name, num_frames):
    """Generate first clip from text prompt"""
    print(f"Generating {num_frames} frame video from text (seed {seed})...")
    g = torch.Generator("cuda").manual_seed(seed)

    video = pipe_t2v(
        prompt=prompt,
        negative_prompt=neg,
        num_inference_steps=STEPS,
        guidance_scale=GUIDE,
        num_frames=num_frames,
        height=H,
        width=W,
        generator=g
    ).frames[0]

    # Convert frames - LTX outputs PIL Images already in uint8 format (0-255)
    frames = [np.array(f) for f in video]
    imageio.mimwrite(name, frames, fps=FPS, quality=9)
    print("wrote", name)

    # Clear GPU memory after generation
    torch.cuda.empty_cache()

    # Return last frame for continuation
    return name, video[-1]

def gen_img2video(prompt, seed, start_frame, name, num_frames):
    """Generate continuation clip from starting frame"""
    print(f"Generating {num_frames} frame video from image (seed {seed})...")
    g = torch.Generator("cuda").manual_seed(seed)

    video = pipe_i2v(
        image=start_frame,
        prompt=prompt,
        negative_prompt=neg,
        num_inference_steps=STEPS,
        guidance_scale=GUIDE,
        num_frames=num_frames,
        height=H,
        width=W,
        generator=g
    ).frames[0]

    # Convert frames
    frames = [np.array(f) for f in video]
    imageio.mimwrite(name, frames, fps=FPS, quality=9)
    print("wrote", name)

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Return last frame for next continuation
    return name, video[-1]


# Generate timestamp for this batch
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
clip_files = []
DURATION = config.get("duration", 10)
last_frame = None

# Generate all clips
for i, clip_config in enumerate(CLIP_CONFIGS, 1):
    seed = random.randint(1, 2 ** 31 - 1)
    filename = f"{OUTDIR}/{SCENE_NAME}_{timestamp}_part{i}.mp4"

    if i == 1:
        # First clip: text-to-video
        filename, last_frame = gen_text2video(prompt, seed, filename, clip_config["frames"])
    else:
        # Subsequent clips: image-to-video continuation
        filename, last_frame = gen_img2video(prompt, seed, last_frame, filename, clip_config["frames"])

    clip_files.append(filename)

    # Save parameters for this clip
    params = {
        "prompt": prompt,
        "negative_prompt": neg,
        "seed": seed,
        "model_id": MODEL_ID,
        "width": W,
        "height": H,
        "fps": FPS,
        "length_seconds": clip_config["length"],
        "num_frames": clip_config["frames"],
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDE,
        "video_quality": 9,
        "part": i,
        "total_parts": len(CLIP_CONFIGS)
    }
    param_file = os.path.splitext(filename)[0] + ".json"
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=2)
    print("wrote", param_file)

# Concatenate all clips using ffmpeg
print("\nConcatenating clips...")
concat_list_name = f"concat_list_{timestamp}.txt"
concat_list = f"{OUTDIR}/{concat_list_name}"
with open(concat_list, 'w') as f:
    for clip_file in clip_files:
        # Use just the filename since we'll run ffmpeg from OUTDIR
        f.write(f"file '{os.path.basename(clip_file)}'\n")

output_file_name = f"{SCENE_NAME}_{timestamp}_{DURATION}sec.mp4"
concat_cmd = [
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", concat_list_name, "-c", "copy", output_file_name
]
subprocess.run(concat_cmd, cwd=OUTDIR, check=True)

print(f"Final video created: {OUTDIR}/{output_file_name}")
print(f"Individual clips saved as: {', '.join([os.path.basename(f) for f in clip_files])}")

# Clean up concat list
os.remove(concat_list)
