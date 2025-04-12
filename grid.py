import os
import random
import torch
import argparse
import concurrent.futures
import gc
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.font_manager as fm
from functools import partial

# For Code A components
from diffusers import StableDiffusionPipeline, DDPMScheduler

# For new Code B components 
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel

# Configure memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Constants for the grid and image generation
NUM_ITERATIONS = 50
NUM_ROWS = 10
IMAGES_PER_ROW = 5
IMAGE_SIZE = (256, 256)  # Size for the comparison grid
GEN_IMAGE_SIZE = 512     # Size for generated images (will be resized for grid)
MAX_WORKERS = 1  # Number of concurrent workers (adjust based on your GPU memory)

def setup_directories():
    """Create necessary directories for outputs"""
    dirs = {
        'code_a_output': './code_a_output',
        'code_b_output': './code_b_output',
        'combined_output': './combined_output',
        'grid_output': './grid_output_nice'
    }
    
    for dir_name, dir_path in dirs.items():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def load_prompt_list(prompt_path):
    """Load prompts from file"""
    try:
        with open(prompt_path, 'r') as file:
            prompts = [line.strip() for line in file.readlines() if line.strip()]
        if not prompts:
            # Fallback prompts if file is empty
            prompts = ["surreal", "fantasy", "abstract", "vibrant", "mysterious"]
    except Exception as e:
        print(f"Error loading prompts from {prompt_path}: {e}")
        # Fallback prompts if file can't be loaded
        prompts = ["surreal", "fantasy", "abstract", "vibrant", "mysterious"]
    
    return prompts

def get_image_paths(pacs_folder):
    """Get all valid image paths from PACS dataset with their domain and class info"""
    styles = ['art_painting', 'cartoon', 'photo', 'sketch']
    objects = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    
    all_paths = []
    
    for style in styles:
        for obj in objects:
            dir_path = os.path.join(pacs_folder, style, obj)
            if not os.path.exists(dir_path):
                continue
            
            # Look for both jpg and png files
            for ext in ['.jpg', '.jpeg', '.png']:
                paths = list(Path(dir_path).glob(f"*{ext}"))
                for path in paths:
                    all_paths.append((str(path), style, obj))
    
    return all_paths

def initialize_code_a_model(args):
    """Initialize just the Code A model with memory optimizations"""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize Code A components
    print("Loading models for Code A (Simple Stable Diffusion)...")
    model_id = args.model_id
    pipe_a = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe_a = pipe_a.to(device)
    
    # Enable optimizations
    pipe_a.enable_attention_slicing(1)
    if hasattr(pipe_a, 'enable_xformers_memory_efficient_attention'):
        pipe_a.enable_xformers_memory_efficient_attention()
    
    return {
        'device': device,
        'pipe_a': pipe_a
    }

def initialize_code_b_model(args):
    """Initialize the new Code B model with memory optimizations"""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize new Code B components
    print("Loading models for Code B (Positive-Negative Embedding Approach)...")
    
    # Load CLIP components first
    feature_extractor = CLIPImageProcessor.from_pretrained(args.clip_path)
    
    # Load Stable Diffusion components
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_path,
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    ).to(device)
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_path,
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(device)
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_path,
        subfolder="unet",
        torch_dtype=torch.float16
    ).to(device)
    
    # Create pipeline with all required components
    pipe_b = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDPMScheduler.from_pretrained(args.pretrained_path, subfolder="scheduler"),
        feature_extractor=feature_extractor,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    
    # Enable optimizations
    pipe_b.enable_attention_slicing(1)
    if hasattr(pipe_b, 'enable_xformers_memory_efficient_attention'):
        pipe_b.enable_xformers_memory_efficient_attention()
    
    # Load CLIP components for image inversion
    clip_model = CLIPModel.from_pretrained(
        args.clip_path,
        torch_dtype=torch.float16
    ).to(device)
    clip_vision_model = clip_model.vision_model.to(device)
    image_processor = CLIPImageProcessor.from_pretrained(args.clip_path)
    
    # Calculate projections for image inversion
    text_proj = clip_model.text_projection.weight.to(device)
    inv_text = torch.linalg.pinv(text_proj.float(), atol=0.3).to(device)
    visual_projection = clip_model.visual_projection.weight.to(device)
    
    # Explicitly delete unused components to free memory
    del clip_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'device': device,
        'pipe_b': pipe_b,
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'clip_vision_model': clip_vision_model,
        'image_processor': image_processor,
        'visual_projection': visual_projection,
        'inv_text': inv_text
    }

def compute_cosine_similarity(img1, img2, image_processor, clip_vision_model, device):
    """
    Computes the cosine similarity between two PIL images using the CLIP image encoder.
    """
    # Convert images to pixel tensors
    img1_tensor = image_processor(img1, return_tensors='pt').pixel_values.to(device)
    img2_tensor = image_processor(img2, return_tensors='pt').pixel_values.to(device)
    
    # Extract features using the CLIP vision model's pooler output
    with torch.inference_mode():
        feat1 = clip_vision_model(pixel_values=img1_tensor).pooler_output
        feat2 = clip_vision_model(pixel_values=img2_tensor).pooler_output
    
    # Normalize the features
    feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity (dot product, since features are normalized)
    cosine_sim = (feat1 * feat2).sum(dim=-1).item()
    return cosine_sim

def generate_image_code_a(models, domain, obj_class, prompt, args, img_index):
    """Generate image using Code A approach"""
    pipe = models['pipe_a']
    device = models['device']
    
    # Crafting a negative prompt similar to the original Code A
    fine_negative_prompt = (
        "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, "
        "morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
        "mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, "
        "disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, "
        "fused fingers, too many fingers, long neck, username, watermark, signature"
    )
    
    # Craft the positive prompt
    pos_prompt = f"{domain.replace('_', ' ')} of {prompt} merged with {obj_class}"
    
    # Generate the image
    with torch.cuda.amp.autocast():
        image = pipe(
            prompt=pos_prompt,
            negative_prompt=fine_negative_prompt,
            guidance_scale=args.guidance_scale,
            height=GEN_IMAGE_SIZE,
            width=GEN_IMAGE_SIZE,
            num_inference_steps=args.num_inference_steps
        ).images[0]
    
    # Create a filename based on attributes
    safe_domain = domain.replace(" ", "_")
    safe_prompt = prompt.replace(" ", "_").replace(",", "")
    safe_class = obj_class.replace(" ", "_")
    filename = f"a_{safe_class}_{safe_domain}_{safe_prompt}_{img_index}.jpg"
    
    # Save the image
    filepath = os.path.join(args.dirs['code_a_output'], filename)
    image.save(filepath)
    
    return image, filepath

def generate_image_code_b(models, image_path, obj_class, prompt, args, img_index):
    """Generate image using new Code B approach (Positive-Negative Embedding)"""
    device = models['device']
    pipe = models['pipe_b']
    tokenizer = models['tokenizer']
    text_encoder = models['text_encoder']
    clip_vision_model = models['clip_vision_model']
    image_processor = models['image_processor']
    visual_projection = models['visual_projection']
    inv_text = models['inv_text']
    
    # Open the original image
    original_image = Image.open(image_path).convert("RGB")
    
    # Extract domain from image_path
    domain = Path(image_path).parts[-3]  # Assuming format: /pacs_data/domain/class/img.jpg
    
    # Prepare the positive prompt
    positive_prompt = f"a {domain} of {prompt}"
    
    # Process positive prompt normally
    pos_inputs = tokenizer(
        positive_prompt, 
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.inference_mode():
        positive_embeddings = text_encoder(pos_inputs.input_ids)[0]
    
    # Process the image to create a negative prompt embedding
    with torch.cuda.amp.autocast():
        # Process the image with CLIP
        clip_image = image_processor(original_image, return_tensors='pt').pixel_values.to(device)
        image_emb = clip_vision_model(pixel_values=clip_image).pooler_output.to(device)
        
        # Project image embedding to text space
        image_emb_proj = (image_emb @ visual_projection.T) @ inv_text.T
        image_emb_proj = image_emb_proj / image_emb_proj.norm(dim=1, keepdim=True)
        
        # Create a "dummy" negative prompt to get the right shape for the negative embeddings
        neg_inputs = tokenizer(
            "dummy negative prompt",
            max_length=tokenizer.model_max_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Get the base text embeddings structure
        negative_embeddings = text_encoder(neg_inputs.input_ids)[0]
        
        # Replace the content with our image embeddings
        # Keep first token as is (usually token position 0 is special), replace others with image embedding
        negative_embeddings[:, 1:] = image_emb_proj.unsqueeze(1)
    
    # Generate image using the pipeline with positive and negative prompts
    with torch.cuda.amp.autocast():
        result = pipe(
            prompt=None,
            prompt_embeds=positive_embeddings,
            negative_prompt_embeds=negative_embeddings,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=GEN_IMAGE_SIZE,
            width=GEN_IMAGE_SIZE
        ).images[0]
    
    # Calculate cosine similarity for informational purposes
    cosine_sim = compute_cosine_similarity(
        original_image, result, 
        image_processor, clip_vision_model, device
    )
    print(f"Cosine similarity between original and generated image: {cosine_sim:.3f}")
    
    # Create a filename based on attributes
    safe_domain = domain.replace(" ", "_")
    safe_prompt = prompt.replace(" ", "_").replace(",", "")
    safe_class = obj_class.replace(" ", "_")
    filename = f"b_{safe_class}_{safe_domain}_{safe_prompt}_{img_index}_sim_{cosine_sim:.3f}.jpg"
    
    # Save the individual image
    filepath = os.path.join(args.dirs['code_b_output'], filename)
    result.save(filepath)
    
    # Create and save a side-by-side comparison
    original_image_resized = original_image.resize((GEN_IMAGE_SIZE, GEN_IMAGE_SIZE))
    composite_width = original_image_resized.width + result.width
    composite_height = result.height  # both images have the same height
    composite = Image.new('RGB', (composite_width, composite_height))
    composite.paste(original_image_resized, (0, 0))
    composite.paste(result, (original_image_resized.width, 0))
    
    composite_filename = f"comp_{safe_class}_{safe_domain}_{safe_prompt}_{img_index}_sim_{cosine_sim:.3f}.jpg"
    composite_filepath = os.path.join(args.dirs['combined_output'], composite_filename)
    composite.save(composite_filepath)
    
    return result, filepath, composite_filepath

def process_code_a_iteration(i, image_paths, prompts, models, args):
    """Process a single Code A iteration"""
    try:
        # Randomly select an image and its metadata
        image_path, domain, obj_class = random.choice(image_paths)
        
        # Randomly select a prompt
        prompt = random.choice(prompts)
        
        print(f"\nWorker processing Code A iteration {i+1}/{NUM_ITERATIONS}")
        print(f"Selected image: {image_path}")
        print(f"Domain: {domain}, Class: {obj_class}")
        print(f"Selected prompt: {prompt}")
        
        # Load original image for the grid
        original_image = Image.open(image_path).convert("RGB")
        
        # Generate image using Code A
        print(f"Worker {i}: Generating image with Code A...")
        code_a_image, _ = generate_image_code_a(models, domain, obj_class, prompt, args, i)
        
        # Return the result with metadata for later use with Code B
        return {
            'index': i,
            'image_path': image_path,
            'domain': domain,
            'obj_class': obj_class,
            'prompt': prompt,
            'original_image': original_image,
            'code_a_image': code_a_image
        }
    except Exception as e:
        print(f"Error in Code A worker {i}: {e}")
        return None

def process_code_b_iteration(result, models, args):
    """Process a single Code B iteration using results from Code A"""
    try:
        i = result['index']
        image_path = result['image_path']
        obj_class = result['obj_class']
        prompt = result['prompt']
        
        print(f"\nWorker processing Code B iteration {i+1}/{NUM_ITERATIONS}")
        print(f"Using image: {image_path}")
        print(f"Selected prompt: {prompt}")
        
        # Generate image using Code B
        print(f"Worker {i}: Generating image with Code B...")
        code_b_image, _, _ = generate_image_code_b(models, image_path, obj_class, prompt, args, i)
        
        # Add Code B result to the existing result dictionary
        result['code_b_image'] = code_b_image
        return result
    except Exception as e:
        print(f"Error in Code B worker {i}: {e}")
        return None

def create_comparison_grid(results, args):
    """Create a grid showing originals and generated images from both approaches"""
    # Sort results by index
    sorted_results = sorted(results, key=lambda x: x['index'])
    
    # Extract images
    original_images = [r['original_image'] for r in sorted_results]
    code_a_images = [r['code_a_image'] for r in sorted_results]
    code_b_images = [r['code_b_image'] for r in sorted_results]
    
    section_gap = 80  # Gap between sections
    header_height = 60  # Height for section headers
    
    # Calculate the image grid dimensions
    total_width = 3 * IMAGES_PER_ROW * IMAGE_SIZE[0]  # 3 columns of images
    
    # Calculate rows for each section
    num_rows = len(original_images) // IMAGES_PER_ROW
    if len(original_images) % IMAGES_PER_ROW != 0:
        num_rows += 1
    
    # Calculate total height with proper spacing
    total_height = (
        header_height +  # Initial header space
        (num_rows * IMAGE_SIZE[1]) +  # Images
        section_gap  # Bottom padding
    )
    
    # Create the grid image
    grid_img = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(grid_img)
    
    # Try to find a suitable font
    try:
        font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # Try to find Arial first, then fall back to any sans-serif font
        arial_paths = [p for p in font_paths if 'arial' in p.lower() or 'liberation' in p.lower() 
                      or 'helvetica' in p.lower() or 'dejavu' in p.lower()]
        
        if arial_paths:
            font_path = arial_paths[0]
        else:
            font_path = font_paths[0]  # Just use the first available font
            
        header_font = ImageFont.truetype(font_path, 30)
    except:
        # Fall back to default font if no TrueType fonts are found
        header_font = ImageFont.load_default()
    
    # Draw headers for each column
    headers = [
        "Original images from PACS dataset",
        "Code A: Simple Stable Diffusion",
        "Code B: Pos-Neg Embedding Approach"
    ]
    
    for i, header in enumerate(headers):
        col_width = IMAGES_PER_ROW * IMAGE_SIZE[0]
        col_start_x = i * col_width
        center_x = col_start_x + col_width // 2 - 150  # Approximate centering
        draw.text((center_x, 10), header, fill="black", font=header_font)
    
    # Calculate starting Y position for images
    images_start_y = header_height
    
    # Place the images in their respective columns
    for idx in range(len(original_images)):
        row = idx // IMAGES_PER_ROW
        col = idx % IMAGES_PER_ROW
        
        # Original image position (first column)
        x1 = col * IMAGE_SIZE[0]
        y1 = row * IMAGE_SIZE[1] + images_start_y
        grid_img.paste(original_images[idx].resize(IMAGE_SIZE), (x1, y1))
        
        # Code A image position (second column)
        x2 = x1 + IMAGES_PER_ROW * IMAGE_SIZE[0]
        grid_img.paste(code_a_images[idx].resize(IMAGE_SIZE), (x2, y1))
        
        # Code B image position (third column)
        x3 = x2 + IMAGES_PER_ROW * IMAGE_SIZE[0]
        grid_img.paste(code_b_images[idx].resize(IMAGE_SIZE), (x3, y1))
    
    # Draw vertical separating lines between columns
    line1_x = IMAGES_PER_ROW * IMAGE_SIZE[0]
    line2_x = 2 * IMAGES_PER_ROW * IMAGE_SIZE[0]
    
    # Draw lines extending throughout the grid
    draw.line([(line1_x, 0), (line1_x, total_height)], fill="black", width=2)
    draw.line([(line2_x, 0), (line2_x, total_height)], fill="black", width=2)
    
    # Save the grid
    output_path = os.path.join(args.dirs['grid_output'], "comparison_grid_new.jpg")
    grid_img.save(output_path, "JPEG", quality=90)
    print(f"Comparison grid saved to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate and compare images using two different methods")
    parser.add_argument("--pacs_folder", type=str, default="./pacs_data", help="Path to PACS dataset")
    parser.add_argument("--prompt_path", type=str, default="./prompts/prompts_list_pacs.txt", help="Path to prompts file")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID for Code A")
    parser.add_argument("--pretrained_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to pretrained model for Code B")
    parser.add_argument("--clip_path", type=str, default="openai/clip-vit-large-patch14", help="Path to CLIP model")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for diffusion")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of diffusion steps")
    parser.add_argument("--edit", action="store_true", default=False, help="Whether to use edit mode for Code B")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for edit mode in Code B")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=MAX_WORKERS, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    # Set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories
    args.dirs = setup_directories()
    
    # Load prompts
    prompts = load_prompt_list(args.prompt_path)
    if not prompts:
        print("Error: No prompts available. Please check the prompt file.")
        return
    
    # Get all image paths from PACS dataset
    image_paths = get_image_paths(args.pacs_folder)
    if not image_paths:
        print(f"Error: No images found in {args.pacs_folder}. Please check the path.")
        return
    
    # ------------------------------------------------------------------------
    # PHASE 1: Initialize and run Code A (Simple Stable Diffusion) for all iterations
    # ------------------------------------------------------------------------
    print("PHASE 1: Running Code A (Simple Stable Diffusion) for all iterations")
    models_a = initialize_code_a_model(args)
    
    code_a_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Process each Code A iteration in parallel
        future_to_idx = {
            executor.submit(
                process_code_a_iteration, 
                i, 
                image_paths, 
                prompts, 
                models_a, 
                args
            ): i for i in range(NUM_ITERATIONS)
        }
        
        # Process results as they complete
        with tqdm(total=NUM_ITERATIONS, desc="Code A Progress") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        code_a_results.append(result)
                except Exception as e:
                    print(f"Error in Code A iteration {idx}: {e}")
                progress_bar.update(1)
    
    # Clean up Code A models to free memory
    print("Cleaning up Code A models...")
    del models_a
    torch.cuda.empty_cache()
    gc.collect()
    
    # ------------------------------------------------------------------------
    # PHASE 2: Initialize and run Code B (Pos-Neg Embedding Approach) for all iterations
    # ------------------------------------------------------------------------
    print("PHASE 2: Running Code B (Pos-Neg Embedding Approach) for all iterations")
    models_b = initialize_code_b_model(args)
    
    code_b_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Process each Code B iteration in parallel using Code A results
        future_to_idx = {
            executor.submit(
                process_code_b_iteration, 
                result, 
                models_b, 
                args
            ): result['index'] for result in code_a_results
        }
        
        # Process results as they complete
        with tqdm(total=len(code_a_results), desc="Code B Progress") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        code_b_results.append(result)
                except Exception as e:
                    print(f"Error in Code B for iteration {idx}: {e}")
                progress_bar.update(1)
    
    # Clean up Code B models to free memory
    print("Cleaning up Code B models...")
    del models_b
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create the comparison grid
    print("Creating comparison grid...")
    grid_path = create_comparison_grid(code_b_results, args)
    
    print("Done! All images generated and comparison grid created.")
    print(f"Check the results in these directories:")
    for dir_name, dir_path in args.dirs.items():
        print(f"- {dir_name}: {dir_path}")

if __name__ == "__main__":
    main()