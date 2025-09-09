import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import glob
import json
from tqdm import tqdm
from pathlib import Path
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Constants from the original notebook ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_ID = "../models/Vintern-1B-v3_5"  # Use local model path
GEN_CFG = dict(max_new_tokens=256, do_sample=False, num_beams=1, repetition_penalty=1.2)

# --- Helper functions ---
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --- Global variables for model/tokenizer, loaded once per process ---
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None

def caption_worker(image_file_list_with_video_name, device_id, question="Mô tả chính, bao gồm text nếu có."):
    """
    Worker function for each process. It loads the model and tokenizer to its assigned GPU
    and then captions the given list of images.
    image_file_list_with_video_name is a list of tuples: (image_path, video_name)
    """
    global GLOBAL_MODEL, GLOBAL_TOKENIZER

    # Load model and tokenizer only once per process
    if GLOBAL_MODEL is None:
        print(f"Process {os.getpid()} loading model on cuda:{device_id}", flush=True)
        GLOBAL_MODEL = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval().to(f'cuda:{device_id}')

        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
        GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
        GLOBAL_MODEL.config.pad_token_id = GLOBAL_TOKENIZER.pad_token_id
        print(f"Process {os.getpid()} model loaded on cuda:{device_id}", flush=True)

    results = []
    current_device = f'cuda:{device_id}'

    # Iterate and caption images for this chunk
    for image_path, video_name in tqdm(image_file_list_with_video_name, desc=f"Captioning on {current_device} (pid:{os.getpid()})", leave=False):
        try:
            # Ensure pixel_values are moved to the correct device for the model
            pixel_values = load_image(image_path, max_num=1).to(torch.bfloat16).to(current_device)
            response = GLOBAL_MODEL.chat(GLOBAL_TOKENIZER, pixel_values, f"<image>\n{question}", GEN_CFG, history=None, return_history=False)
            results.append({'image_path': str(image_path), 'video_name': video_name, 'caption': response})
        except Exception as e:
            print(f"Error processing {image_path} on {current_device}: {e}", flush=True)
            results.append({'image_path': str(image_path), 'video_name': video_name, 'caption': f"ERROR: {e}"})
    return results

def main():
    # Create the output directory if it doesn't exist
    output_dir = Path('./captioning')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all image paths from resource folder - check multiple possible paths
    possible_paths = [
        Path('../resources/keyframes'),
    ]
    
    base_dir = None
    for path in possible_paths:
        if path.exists():
            base_dir = path
            print(f"Found resources at: {base_dir.absolute()}")
            break
    
    if base_dir is None:
        print("Error: Could not find resources directory in any of these locations:")
        for path in possible_paths:
            print(f"  - {path.absolute()}")
        return

    all_images_to_process = []
    
    # Look for images directly in resource folder and subdirectories
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    for ext in image_extensions:
        for image_path in base_dir.rglob(ext):
            # Try to determine video name from folder structure
            video_name = image_path.parent.name
            # If parent is 'keyframes', use grandparent as video name
            if video_name.lower() == 'keyframes':
                video_name = image_path.parent.parent.name
            all_images_to_process.append((image_path, video_name))

    if not all_images_to_process:
        print("No images found in resource directory!")
        print(f"Searched in: {base_dir.absolute()}")
        return
    
    # Sort images to ensure consistent distribution
    all_images_to_process.sort(key=lambda x: (x[1], x[0]))
    all_images_to_process = all_images_to_process[:2]
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs detected. Running on CPU with 1 process. This will be very slow.", flush=True)
        num_gpus = 1

    print(f"Total images to process: {len(all_images_to_process)}", flush=True)
    print(f"Detected {num_gpus} GPUs. Using {num_gpus} processes for parallel captioning.", flush=True)

    # Split image paths evenly among processes
    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(all_images_to_process):
        chunks[i % num_gpus].append(item)

    results_from_all_processes = []
    # Use ProcessPoolExecutor to manage parallel execution on GPUs
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i in range(num_gpus):
            # Submit a task for each chunk, passing the chunk and its assigned device_id
            futures.append(executor.submit(caption_worker, chunks[i], i))

        # Use tqdm to show overall progress as futures complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall Captioning Progress"):
            results_from_all_processes.extend(future.result())

    # Write individual JSON files for each keyframe
    print("\nWriting individual JSON files for each keyframe...", flush=True)
    
    # Create output directory structure similar to resources
    for item in tqdm(results_from_all_processes, desc="Writing JSON files"):
        image_path = Path(item['image_path'])
        video_name = item['video_name']
        frame_name = image_path.stem  # Get filename without extension
        caption = item['caption']
        
        # Create video-specific output directory
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create JSON structure similar to the example
        json_data = {
            "caption": caption,
        }
        
        # Save as individual JSON file
        json_file = video_output_dir / f"{frame_name}.json"
        with open(json_file, mode='w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Captioning complete! Individual JSON files saved to {output_dir}")
    print(f"Total files processed: {len(results_from_all_processes)}")

if __name__ == "__main__":
    main()