import argparse
import os
import sys
import random
from glob import glob
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPImageProcessor
from huggingface_hub import login

# Configure memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def compute_cosine_similarity(img1, img2, image_processer, clip_vision, device):
    """Computes CLIP cosine similarity between two images."""
    img1_tensor = image_processer(img1, return_tensors='pt').pixel_values.to(device)
    img2_tensor = image_processer(img2, return_tensors='pt').pixel_values.to(device)
    feat1 = clip_vision(pixel_values=img1_tensor).pooler_output
    feat2 = clip_vision(pixel_values=img2_tensor).pooler_output
    return torch.nn.functional.cosine_similarity(feat1, feat2).item()

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Authentication
    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    try:
        # Load img2img pipeline directly
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.pretrained_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Enable optimizations
        pipe.enable_attention_slicing(1)
        pipe.enable_xformers_memory_efficient_attention()

    except Exception as e:
        print(f"Pipeline loading failed: {str(e)}")
        sys.exit(1)

    # Load CLIP for similarity calculation
    try:
        clip_model = CLIPModel.from_pretrained(args.clip_path, torch_dtype=torch.float16).to(device)
        clip_vision = clip_model.vision_model
        image_processer = CLIPImageProcessor.from_pretrained(args.clip_path)
    except Exception as e:
        print(f"CLIP loading failed: {str(e)}")
        sys.exit(1)

    # Configuration
    styles = ['art_painting', 'photo', 'cartoon', 'sketch']
    classes = ['guitar', 'house', 'person', 'elephant', 'dog', 'giraffe', 'horse']
    os.makedirs(args.output_dir, exist_ok=True)
    
    cosine_sim_list = []
    
    for i in range(50):
        # Random selection
        style = random.choice(styles)
        obj_class = random.choice(classes)
        img_dir = os.path.join('./pacs_data', style, obj_class)
        
        # Find and load image
        ext = 'png' if style == "sketch" else 'jpg'
        images = glob(os.path.join(img_dir, f"*.{ext}"))
        if not images:
            print(f"No images found in {img_dir}, skipping iteration {i}")
            continue
            
        img_path = random.choice(images)
        original_img = Image.open(img_path).convert("RGB")
        resized_img = original_img.resize((args.resolution, args.resolution))

        # Generate with img2img
        result = pipe(
            prompt="a random image",  # Empty prompt
            image=resized_img,
            strength=0.7,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps
        ).images[0]

        # Calculate similarity
        sim = compute_cosine_similarity(resized_img, result, image_processer, clip_vision, device)
        cosine_sim_list.append(sim)

        # Save results
        composite = Image.new('RGB', (resized_img.width * 2, resized_img.height))
        composite.paste(resized_img, (0, 0))
        composite.paste(result, (resized_img.width, 0))
        composite.save(os.path.join(args.output_dir, f"{style}_{obj_class}_cosine_{sim:.3f}_{i}.jpg"))

        if i % 10 == 0:
            torch.cuda.empty_cache()

    if cosine_sim_list:
        avg_sim = sum(cosine_sim_list) / len(cosine_sim_list)
        print(f"Average CLIP similarity: {avg_sim:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--clip_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--output_dir", type=str, default="./img2img_results")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    args = parser.parse_args()
    main(args)