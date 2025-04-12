import argparse
import os
import sys
import random
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPImageProcessor
from prompt_clip import CLIPVisionModelWithPrompt
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from huggingface_hub import login

# Configure memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def compute_cosine_similarity(img1, img2, image_processer, clip_for_inverse_matrix, device):
    """
    Computes the cosine similarity between two PIL images using the CLIP image encoder.
    """
    # Convert images to pixel tensors
    img1_tensor = image_processer(img1, return_tensors='pt').pixel_values.to(device)
    img2_tensor = image_processer(img2, return_tensors='pt').pixel_values.to(device)
    # Extract features using the CLIP vision model's pooler output
    feat1 = clip_for_inverse_matrix(pixel_values=img1_tensor).pooler_output
    feat2 = clip_for_inverse_matrix(pixel_values=img2_tensor).pooler_output
    # Normalize the features
    feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    # Compute cosine similarity (dot product, since features are normalized)
    cosine_sim = (feat1 * feat2).sum(dim=-1).item()
    return cosine_sim

def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Authentication handling
    if os.environ.get("HF_TOKEN"):
        try:
            login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
        except ValueError as e:
            print("ERROR: Invalid Hugging Face token!")
            sys.exit(1)

    try:
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
        pipe = StableDiffusionPipeline(
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
        pipe.enable_attention_slicing(1)
        pipe.enable_xformers_memory_efficient_attention()

    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        sys.exit(1)

    # -------------------------------
    # Load CLIP components for image inversion
    # -------------------------------
    try:
        clip_model = CLIPModel.from_pretrained(
            args.clip_path,
            torch_dtype=torch.float16
        ).to(device)
        clip_for_inverse_matrix = clip_model.vision_model.to(device)
        image_processer = CLIPImageProcessor.from_pretrained(args.clip_path)
    except Exception as e:
        print(f"CLIP loading failed: {str(e)}")
        sys.exit(1)

    # Calculate projections for image inversion
    #with torch.inference_mode():
    text_proj = clip_model.text_projection.weight.to(device)
    inv_text = torch.linalg.pinv(text_proj.float(), atol=0.3).to(device)
    visual_projection = clip_model.visual_projection.weight.to(device)
    del clip_model

    # -------------------------------
    # Define the directory structure for image selection
    # -------------------------------
    styles = ['art_painting', 'photo', 'cartoon', 'sketch']
    classes = ['guitar', 'house', 'person', 'elephant', 'dog', 'giraffe','horse']
    closed_set_object = random.choice(classes)  # Using guitar as the closed set object
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------
    # Load prompt list from file
    # -------------------------------
    prompt_list_path = "./prompts/prompts_list_pacs.txt"
    try:
        with open(prompt_list_path, "r") as f:
            prompt_list = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Prompt list loading failed: {str(e)}")
        sys.exit(1)

    cosine_sim_list = []  # Will store cosine similarity values for all iterations
    
    # -------------------------------
    # Iterate 50 times, choosing random images and prompts each time
    # -------------------------------
    for i in range(50):
        # Randomly select style but always use the closed set object (guitar)
        random_style = random.choice(styles)
        image_dir = os.path.join('./pacs_data', random_style, closed_set_object)
        if random_style == "sketch":
            img_list = glob(os.path.join(image_dir, "*.png"))
        else:    
            img_list = glob(os.path.join(image_dir, "*.jpg"))
        
        if len(img_list) == 0:
            print(f"No images found in {image_dir}, skipping iteration {i}.")
            continue
        
        chosen_img = random.choice(img_list)
        print(f"Iteration {i}: Selected image from style '{random_style}', object '{closed_set_object}'; Path: {chosen_img}")

        # Open the original image
        original_image = Image.open(chosen_img).convert("RGB")
        
        # -------------------------------
        # Select a random prompt from file for positive prompt
        # -------------------------------
        random_prompt = random.choice(prompt_list)
        # Note that we're not mixing with banana in the positive prompt
        positive_prompt = f"a {random_style} of {random_prompt}"
        
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
        
        # -------------------------------
        # Process the image to create a negative prompt embedding
        # -------------------------------
        with torch.cuda.amp.autocast():
            # Process the image with CLIP
            clip_image = image_processer(original_image, return_tensors='pt').pixel_values.to(device)
            image_emb = clip_for_inverse_matrix(pixel_values=clip_image).pooler_output.to(device)
            
            # Project image embedding to text space
    
            image_emb_proj = (image_emb @ visual_projection.T) @ inv_text.T
            image_emb_proj = image_emb_proj / image_emb_proj.norm(dim=1, keepdim=True)
            
            # Create a "dummy" negative prompt to get the right shape for the negative embeddings
            # We'll replace the actual embeddings with our image-derived ones
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

        # -------------------------------
        # Generate image using the pipeline with positive and negative prompts
        # -------------------------------
        with torch.cuda.amp.autocast():
            result = pipe(
                prompt=None,
                prompt_embeds=positive_embeddings,
                negative_prompt_embeds=negative_embeddings,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.resolution,
                width=args.resolution
            ).images[0]
        
        # -------------------------------
        # Compute cosine similarity between original and generated image
        # -------------------------------
        cosine_sim = compute_cosine_similarity(original_image, result, image_processer, clip_for_inverse_matrix, device)
        cosine_sim_list.append(cosine_sim)
        
        # -------------------------------
        # Stack the original and the modified images side by side
        # -------------------------------
        original_image_resized = original_image.resize(result.size)
        composite_width = original_image_resized.width + result.width
        composite_height = result.height  # both images have the same height
        composite = Image.new('RGB', (composite_width, composite_height))
        composite.paste(original_image_resized, (0, 0))
        composite.paste(result, (original_image_resized.width, 0))
        
        # Save composite image with cosine similarity appended to the filename
        safe_prompt = random_prompt.replace(" ", "_")
        output_filename = f"{random_style}_{closed_set_object}_to_{safe_prompt}_cosine_{cosine_sim:.3f}_{i}.jpg"
        composite.save(os.path.join(args.output_dir, output_filename))
        print(f"Saved composite image: {output_filename} in {args.output_dir}")
        print(f"Positive prompt: '{positive_prompt}', Image used as negative prompt")
        
        # Cleanup GPU cache occasionally
        if i % 10 == 0:
            torch.cuda.empty_cache()
            
    # Print average cosine similarity after all iterations
    if cosine_sim_list:
        avg_cosine_sim = sum(cosine_sim_list) / len(cosine_sim_list)
        print(f"Average cosine similarity over {len(cosine_sim_list)} images: {avg_cosine_sim:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="an image of a balloon")
    parser.add_argument("--clip_prompt_length", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument("--onlyprompt", action="store_true", default=False)
    parser.add_argument("--edit", action="store_true", default=False)  # Changed to False as we're not using this approach
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--clip_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./similar")
    parser.add_argument("--seed", type=int, default=10)
    
    args = parser.parse_args()
    main(args)