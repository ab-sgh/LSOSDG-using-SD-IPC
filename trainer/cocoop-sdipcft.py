import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
from typing import List, Dict
import einops
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.linalg

from clip import clip
import einops
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from diffusers import StableDiffusionPipeline
import logging
import random

_tokenizer = _Tokenizer()
df = pd.DataFrame()
cls_label = pd.DataFrame()

# Set the device at the top
device = "cuda" if torch.cuda.is_available() else "cpu"
clip1, preprocess = clip.load("ViT-B/32", device=device)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, feature=None):
        x = prompts.to(device) + self.positional_embedding.type(self.dtype).to(device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if feature is not None:
            feature = einops.repeat(feature, 'm n -> k m n', k=5)
            x[:5, :, :] = x[:5, :, :] + feature
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # Use the index of the end-of-text token (assumed to be the argmax)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

# Updated resize_transform: now it works directly on PIL images.
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class StableDiffusion(nn.Module):
    def __init__(self):
        super(StableDiffusion, self).__init__()
        self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        # Disable safety checker:
        def disabled_safety_checker(images, clip_input):
            if len(images.shape) == 4:
                num_images = images.shape[0]
                return images, [False] * num_images
            else:
                return images, False
        self.pipe.safety_checker = disabled_safety_checker

    def forward(self, batch, pos_prompt, neg_prompt):
        generated_images = []
        # Use a simple batch size logic:
        batchsize = 2 if batch == 5 else 1
        positive_prompt = [pos_prompt] * batchsize
        neg_prompts = [neg_prompt] * batchsize
        with torch.no_grad():
            for i in range(batchsize):
                batch_output = self.pipe(prompt=positive_prompt[i], negative_prompt=neg_prompts[i], guidance_scale=15)
                generated_images.append(batch_output.images[0])
        # Now, since the generated images are PIL Images, we directly apply our resize_transform
        generated_images = torch.stack([resize_transform(img) for img in generated_images]).to(device)
        return generated_images

    def generate_from_embedding(self, text_embeddings, neg_prompt, batch):
        outputs = self.pipe(prompt_embeds=text_embeddings, negative_prompt=neg_prompt, guidance_scale=15)
        return outputs.images

class SDIPCGenerateUnknownImages(nn.Module):
    def __init__(self, clip_model, ft_lambda=0.1, feature_loss_weight=1.0):
        """
        Implements SD-IPC-FT for generating pseudo open images.
        
        Args:
            clip_model: The CLIP model instance.
            ft_lambda: Weight for the regularization loss on the fine-tuning offset.
            feature_loss_weight: Weight for the feature consistency loss.
        """
        super(SDIPCGenerateUnknownImages, self).__init__()
        self.clip_model = clip_model
        self.diffusion = StableDiffusion()  # Using the StableDiffusion instance already in use.
        # Compute the pseudo-inverse of the CLIP text projection (fixed).
        self.W_t_inv = torch.linalg.pinv(clip_model.text_projection.detach().float()).to(device)
        self.constant = 27.0  # Fixed norm constant as in SD-IPC.
        # Up-projection layer: maps from 512 (CLIP text space) to 768 (diffusion model's text embedding space).
        self.up_proj = nn.Linear(self.W_t_inv.shape[0], 768)
        # Trainable deep prompt offset for fine-tuning (SD-IPC-FT).
        self.delta = nn.Parameter(torch.zeros(1, 768))
        self.ft_lambda = ft_lambda
        self.feature_loss_weight = feature_loss_weight

    def forward(self, reference_image, neg_prompt, compute_loss=False):
        """
        Args:
            reference_image: A reference image tensor (or a single image that will be unsqueezed to batch size 1).
            neg_prompt: A negative prompt string for image generation.
            compute_loss (bool): If True, computes and returns the fine-tuning loss.
            
        Returns:
            generated_images_tensor: A tensor of images generated from the pseudo prompt.
            loss_ft: The combined fine-tuning loss (regularization + feature consistency); zero if compute_loss is False.
        """
        # Ensure the reference image has a batch dimension.
        if reference_image.dim() == 3:
            reference_image = reference_image.unsqueeze(0)
        
        # Compute the visual embedding using the CLIP visual encoder (frozen).
        with torch.no_grad():
            f_img = self.clip_model.visual(reference_image.to(device).float())
        if isinstance(f_img, tuple):
            f_img = f_img[0]
        # Normalize the visual embedding.
        f_img_norm = f_img / f_img.norm(dim=-1, keepdim=True)
        f_img_norm = f_img_norm.float()
        
        # Closed-form inversion: compute the initial pseudo prompt embedding (in 512 dimensions).
        f_prompt_initial = self.constant * (self.W_t_inv @ f_img_norm.T).T  # shape: (batch, 512)
        # Up-project to 768 dimensions.
        f_prompt_up = self.up_proj(f_prompt_initial)  # shape: (batch, 768)
        
        # Apply the trainable deep prompt offset (fine-tuning).
        f_prompt_ft = f_prompt_up + self.delta  # shape: (batch, 768)
        
        # Compute regularization loss on the offset.
        loss_ft = self.ft_lambda * torch.norm(self.delta, p=2)**2
        
        # Replicate the refined embedding to form a full prompt sequence (length L, e.g., 77 tokens).
        L = 77
        pseudo_prompt_embedding = f_prompt_ft.unsqueeze(1).repeat(1, L, 1)
        
        # Generate images using the diffusion pipeline that accepts text embeddings.
        generated_images = self.diffusion.generate_from_embedding(
            pseudo_prompt_embedding, neg_prompt, batch=reference_image.shape[0]
        )
        
        # Convert the list of PIL images to a tensor by applying the resize transform and stacking.
        generated_images_tensor = torch.stack([resize_transform(img) for img in generated_images]).to(device)
        
        # Optionally compute a feature consistency loss: ensure generated image features stay close to the reference.
        if compute_loss:
            f_generated = self.clip_model.visual(generated_images_tensor.to(device).float())
            if isinstance(f_generated, tuple):
                f_generated = f_generated[0]
            f_generated_norm = f_generated / f_generated.norm(dim=-1, keepdim=True)
            # Mean-squared error loss between generated image features and the original reference features (detached).
            feature_loss = self.feature_loss_weight * F.mse_loss(f_generated_norm, f_img_norm.detach())
            loss_ft = loss_ft + feature_loss
        else:
            loss_ft = torch.tensor(0.0, device=device)
        
        return generated_images_tensor, loss_ft




class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.15)

    def forward(self, image_features, attribute_embeddings, mask_embed):
        attn_output, attn_weights = self.multihead_attn(image_features, attribute_embeddings, attribute_embeddings, key_padding_mask=mask_embed)
        return attn_output

# Taken from MCAN model:
class AttFlat(nn.Module):
    def __init__(self):
        super(AttFlat, self).__init__()
        self.mlp = MLP(
            in_size=512,
            mid_size=512,
            out_size=1,
            dropout_r=0.15,
            use_relu=True
        )
        self.linear_merge = nn.Linear(512, 512)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(1):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)
    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)
    def forward(self, x):
        return self.linear(self.fc(x))

############################################################################################################
class PL(nn.Module):
    def __init__(self, classnames, clip_model, ntx, config):
        """
        Args:
            classnames (List[str]): List of class names.
            clip_model (nn.Module): Loaded CLIP model.
            ntx (int): Number of context tokens.
            config (dict): Configuration dictionary with keys like "project_dim", "dropout", etc.
        """
        super().__init__()
        n_cls = len(classnames)
        n_ctx = ntx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Create and register context vectors (this defines self.ctx)
        ctx_vectors = torch.empty(n_ctx, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # Define an MLP that maps the raw context to the target context dimension
        self.prompt_cls = nn.Sequential(
            nn.Linear(768, config["project_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["project_dim"], ctx_dim)
        )

        #self.prompt_cls = self.prompt_cls.half()


        # Process the class names:
        # Replace underscores with spaces
        classnames = [name.replace("_", " ") for name in classnames]
        # Calculate the lengths (if needed later)
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # Create a prompt prefix string using "X" repeated n_ctx times
        prompt_prefix = " ".join(["X"] * n_ctx)
        # Construct full prompt strings for each class
        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        # Tokenize prompts using CLIP's tokenizer
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # already on device
        self.name_lens = name_lens


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """
        Concatenate prefix, context and suffix to form the full prompts.
        """
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        # Retrieve the fixed prefix and suffix from buffers
        prefix = self.token_prefix
        suffix = self.token_suffix
        # Process the learnable context tokens through the MLP
        ctx = self.prompt_cls(self.ctx)
        # Expand the processed context to match the number of classes
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # Construct the full prompt tokens
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts, self.ctx
    
class CustomCLIP(nn.Module):
    def __init__(self, classnames: List[str], domainnames: List[str], clip_model: nn.Module, config):
        super().__init__()
        self.ctx = config["n_ctx"]
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.promptlearner = PL(classnames, clip_model, self.ctx, config)
        self.projector = nn.Linear(768, 512)
        self.cross_attention = CrossAttention(512, config["n_head"])
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_class = len(classnames)
        self.classnames = classnames
        self.domainnames = domainnames

    def forward(self, image: torch.Tensor, attri: torch.Tensor, mask_embed: torch.Tensor, label: torch.Tensor = None, dom_label: torch.Tensor = None, batch=None):
        prompts, ctx = self.promptlearner()
        # Ensure that prompts is on the same device as image
        prompts = prompts.to(image.device)
        image_features, _, _ = self.image_encoder(image.type(self.dtype), ctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompt_prefix = " ".join(["X"] * self.ctx)
        tokenized_prompts = torch.cat([clip.tokenize(f"{prompt_prefix} {p}") for p in self.classnames]).to(image.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = self._compute_logits(text_features, image_features, logit_scale)
        margin_loss = None
        if dom_label is not None:
            txt_features = text_features[:-1, :].repeat(image.size(0) - batch, 1)
            loss_sty, sty_embedding = self._compute_style_loss(image_features[:-batch, :], dom_label[:-batch], label[:-batch], attri, mask_embed, logit_scale)
            return logits, loss_sty, None, margin_loss, txt_features, sty_embedding
        else:
            return logits, image_features

    def _compute_text_features(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> List[torch.Tensor]:
        return [self.text_encoder(pts_i, tokenized_prompts).norm(dim=-1, keepdim=True) for pts_i in prompts]

    def _compute_logits(self, txt_features: torch.Tensor, image_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        return logit_scale * image_features @ txt_features.t()

    def _compute_embed_features(self, embed: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        features = self.text_encoder(embed, tokenized_prompts)
        return features / features.norm(dim=-1, keepdim=True)

    def _compute_margin_loss(self, logits: torch.Tensor) -> torch.Tensor:
        soft_logits = F.softmax(logits, dim=1)
        max_known = torch.sum(soft_logits[:, :54], dim=1)[0]
        unknown = soft_logits[:, 54]
        margin = torch.abs(max_known - unknown) - 0.5
        return -torch.min(margin, torch.zeros_like(margin)).mean()
    
    #@torch.cuda.amp.autocast()
    def _compute_style_loss(self, image_features: torch.Tensor, dom_label: torch.Tensor, label: torch.Tensor, attri: torch.Tensor, mask_embed: torch.Tensor, logit_scale: torch.Tensor) -> tuple:
        device = image_features.device
        sty_embedding_list = []
        t = 0
        loss_sty = 0
        for domain in [0, 1, 2]:
            domain_mask = dom_label == domain
            domain_features = image_features[domain_mask]
            original_indices = torch.nonzero(domain_mask, as_tuple=False).squeeze(1)
            domain_labels = label[domain_mask]
            if domain_features.size(0) == 0:
                continue
            t += 1
            d = self.domainnames[domain]
            tokenized_prompts = torch.cat([clip.tokenize(f"A {d.replace('_', ' ')} of a {p}") for p in self.classnames[:-1]]).to(device)
            n_cls = len(self.classnames)
            domain_img = einops.repeat(domain_features, 'm n -> k m n', k=n_cls-1)
            cross_atten = self.cross_attention(
                domain_img, 
                self.projector(attri.to(device)), 
                mask_embed.to(device)
            )
            with torch.no_grad():
                embedding = clip1.token_embedding(tokenized_prompts).type(self.dtype).to(device)
            domain_embeddings = []
            domain_logits = []
            for i in range(domain_features.size(0)):
                embedding_copy = embedding.clone()
                ctx_i = cross_atten[:, i, :].unsqueeze(1)
                embedding_copy[:, 1:5, :] += ctx_i
                embedding_int = self.text_encoder(embedding_copy, tokenized_prompts)
                embedding_int = F.normalize(embedding_int, dim=-1)
                logit = logit_scale * domain_features[i] @ embedding_int.t()
                domain_logits.append(logit)
                domain_embeddings.append(embedding_int)
            domain_logits = torch.stack(domain_logits)
            domain_embeddings = torch.stack(domain_embeddings)
            for i, idx in enumerate(original_indices):
                sty_embedding_list.append((idx.item(), domain_embeddings[i]))
            loss_sty += F.cross_entropy(domain_logits, domain_labels)
        loss_sty /= t
        sty_embedding_list.sort(key=lambda x: x[0])
        sty_embedding = torch.cat([x[1] for x in sty_embedding_list])
        return loss_sty, sty_embedding
