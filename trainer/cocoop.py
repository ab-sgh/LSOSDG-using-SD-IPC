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

from clip import clip
import einops
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from diffusers import StableDiffusionPipeline
import logging
import random
_tokenizer = _Tokenizer()
df = pd.DataFrame()
cls_label = pd.DataFrame()

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

    def forward(self, prompts, tokenized_prompts,feature=None):
        x = prompts.to(device) + self.positional_embedding.type(self.dtype).to(device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if feature is not None:
            feature = einops.repeat(feature, 'm n -> k m n',k=5)
            x[:5,:,:] = x[:5,:,:] + feature
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class StableDiffusion(nn.Module):
    def __init__(self):
        super(StableDiffusion, self).__init__()
        self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        logging.basicConfig(level=logging.WARNING)
        logging.basicConfig(level=logging.WARNING)
        def disabled_safety_checker(images, clip_input):
            if len(images.shape)==4:
                    num_images = images.shape[0]
                    return images, [False]*num_images
            else:
                    return images, False
        self.pipe.safety_checker = disabled_safety_checker
    def forward(self, batch, pos_prompt, neg_prompt):       
        generated_images = []
        # if int(batch*0.1) > 0:
        #     batchsize = int(batch*0.1)
        # else:
        #     batchsize = 1
        if batch ==5:
            batchsize=2
        else:
            batchsize=1
        positive_prompt = [pos_prompt] * batchsize
        neg_prompts = [neg_prompt]  * batchsize
        # negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

        with torch.no_grad():
            for i in range(batchsize):
                batch_output = self.pipe(prompt=positive_prompt[i],negative_prompt=neg_prompts[i],guidance_scale=15)
                generated_images.append(batch_output.images[0])
        generated_images = torch.stack([ToTensor()(img) for img in generated_images]).to(device)
        return generated_images
    
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
  
class GenerateUnknownImages(nn.Module):
    def __init__(self):
        super().__init__()

        self.diffusion = StableDiffusion()

    def forward(self, batch, pos_prompt, neg_prompt):
        '''
        Stable diffusion
        '''

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]      
        normalize = transforms.Normalize(mean=mean, std=std)
        
        generated_unknown_images = self.diffusion(batch, pos_prompt, neg_prompt) 
        resized_unknown_images = torch.stack([resize_transform(x) for x in generated_unknown_images])
        normalized_unknown_images = normalize(resized_unknown_images)
        normalized_unknown_images = normalized_unknown_images.to(device)

        return normalized_unknown_images


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True,dropout=0.15)

    def forward(self, image_features, attribute_embeddings,mask_embed):
        attn_output, attn_weights = self.multihead_attn(image_features, attribute_embeddings, attribute_embeddings,key_padding_mask=mask_embed)
        return attn_output
    
#################### Taken from MCAN model ##################################################3
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

        self.linear_merge = nn.Linear(
            512,
            512
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            # x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            x_mask.unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(1):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

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
    def __init__(self, classnames, clip_model,ntx,config):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = ntx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_vectors = torch.empty(n_ctx, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)

        # ctx_vectors_unk = torch.empty(2, ctx_dim, dtype=dtype)
        # nn.init.normal_(ctx_vectors_unk, std=0.02)
        prompt_prefix = " ".join(["X"] * (n_ctx)) #48
        self.prompt_cls = nn.Sequential(
                                nn.Linear(768, config["project_dim"]),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=config["dropout"]),  # Dropout layer with a probability of 0.5
                                nn.Linear(config["project_dim"], ctx_dim)
                            )
        # self.prompt_cls = nn.Sequential(
        #                             nn.Linear(768, config["project_dim"]),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(config["project_dim"], ctx_dim)
        #                         )
        # self.prompt_cls = nn.Linear(768, ctx_dim)
        # self.feat = nn.Sequential(
        #                             nn.Linear(768, 48),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(48, ctx_dim)
        #                         )

        self.ctx = nn.Parameter(ctx_vectors)
        # self.ctx_k = nn.Parameter(ctx_vectors_unk)
        # self.projector =  nn.Sequential(
        #                             nn.Linear(vis_dim, vis_dim // 16),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(vis_dim // 16, ctx_dim)
        #                         )
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :]) 
        self.register_buffer("token_suffix", embedding[:,  1+n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  
                ctx,   
                suffix, 
            ],
            dim=1,
        )

        return prompts
    def create_mask(self,param, dropout_rate):
            mask = (torch.rand_like(param) > dropout_rate).float()
            return mask
    def apply_mask(self,param, mask):
        return param * mask
    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.prompt_cls(self.ctx)
        # ctx_k = self.ctx_k
        # ctx = torch.cat((ctx_k,ctx),dim=0)
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        # if dom_label is not None:
        #     mask = self.create_mask(ctx,0.25)
        #     ctx =  self.apply_mask(ctx,mask)
        # embed_ctx = torch.cat((ctx.unsqueeze(0).expand(self.n_cls-1, -1, -1),self.ctx_unk.unsqueeze(0)),dim=0)
        # data = self.projector(data) # (b,dim) 
        # ctx_shifted = ctx.unsqueeze(0) # (b, ncls, dim)
        # ctx = ctx.unsqueeze(0).expand(data.size(0),-1,-1)
        # print(cls.shape)
        # print(feat_token.shape)
        # prompts = []
        # for i in range(data.size(0)):
        #     # ctx_i = torch.cat((data[i].unsqueeze(0),ctx))
        #     ctx_i = torch.cat((ctx,self.prompt_cls(cls[i].unsqueeze(0)),self.prompt_cls(feat_token[i].unsqueeze(0))),dim=0)
        #     # print(ctx_i.shape)
        #     ctx_i = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)
        #     # print(ctx_i.shape)
        #     # pts_ctx = torch.cat((ctx_i,self.ctx_unk.unsqueeze(0)),dim=0)
        #     pts_i = self.construct_prompts(ctx_i, prefix, suffix)
        #     # pts_i[self.n_cls-1,:,:] = torch.cat((pts_i[self.n_cls-1,:4,:],ctx_unk,pts_i[self.n_cls-1,5:,:]),dim=0) 
        #     prompts.append(pts_i)
        # prompts = torch.stack(prompts)  
        # print(prompts.shape)    
        return prompts,self.ctx
class CustomCLIP(nn.Module):
    def __init__(self, classnames: List[str], domainnames: List[str], clip_model: nn.Module,config):
        super().__init__()
        self.ctx = config["n_ctx"]
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.promptlearner = PL(classnames, clip_model,self.ctx,config)
        # self.text_encoder_1 = TextEncoder(clip_model)
        self.projector = nn.Linear(768, 512)
        # self.projector = nn.Sequential(
        #                              nn.Linear(768, 32),
        #                              nn.ReLU(inplace=True),
        #                              nn.Linear(32, 512)
        #                          )    
        # self.projector1 = nn.Linear(512, 512)
        # self.projector1 = nn.Sequential(
        #                             nn.Linear(512, 32),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(32, 512)
        #                         )     
        self.cross_attention = CrossAttention(512, config["n_head"]) #2
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_class = len(classnames)
        self.classnames = classnames
        self.domainnames = domainnames

    def forward(self, image: torch.Tensor, attri: torch.Tensor, mask_embed: torch.Tensor, label: torch.Tensor = None, dom_label: torch.Tensor = None,batch=None):
        # with torch.no_grad():

        prompts,ctx = self.promptlearner()
        image_features,_,_ = self.image_encoder(image.type(self.dtype),ctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # with torch.no_grad():
        #     image_features1,_,_ = self.image_encoder(image.type(self.dtype))
        #     image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
        # cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        # score = cos(image_features,image_features1)
        # score = 1.0-torch.mean(score)
        score = None
        prompt_prefix = " ".join(["X"] *self.ctx)
        tokenized_prompts = torch.cat([clip.tokenize(f"{prompt_prefix} {p}") for p in self.classnames]).to(image.device)

        # txt_features = self._compute_text_features(prompts, tokenized_prompts)

        # txt_features = []
        # for pts_i in prompts:
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # txt_features.append(text_features)
        

        logit_scale = self.logit_scale.exp()
        logits = self._compute_logits(text_features, image_features, logit_scale)

        # txt_features = einops.repeat(text_features,'m n -> k m n',k=image.size(0))
        # print(text_features.shape)
        # print(image.size(0))
        # print("txt ",txt_features.shape)

        # print(logits.shape)
        # print(txt_features.shape)
        # text_features = self._compute_embed_features(embed, tokenized_prompts)

        # margin_loss = self._compute_margin_loss(logits)
        margin_loss = None
        # print(image.size(0))
        # print(batch)
        # print(image.size(0)-(3*batch))

        if dom_label is not None:
            txt_features = text_features[:-1,:].repeat(image.size(0)-batch, 1)
            # loss_sty =None
            # sty_embedding = None
            # with torch.no_grad():
            #     image_features1,_,_ = self.image_encoder(image.type(self.dtype),ctx)
            #     image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
            loss_sty, sty_embedding = self._compute_style_loss(image_features[:-batch,:], dom_label[:-batch], label[:-batch], attri, mask_embed, logit_scale)
            return logits, loss_sty,score, margin_loss, txt_features, sty_embedding
        else:
            return logits,image_features

    def _compute_text_features(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> List[torch.Tensor]:
        return [self.text_encoder(pts_i, tokenized_prompts).norm(dim=-1, keepdim=True) for pts_i in prompts]

    def _compute_logits(self, txt_features: List[torch.Tensor], image_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        return  logit_scale * image_features @ txt_features.t()

    def _compute_embed_features(self, embed: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        features = self.text_encoder(embed, tokenized_prompts)
        return features / features.norm(dim=-1, keepdim=True)

    def _compute_margin_loss(self, logits: torch.Tensor) -> torch.Tensor:
        soft_logits = F.softmax(logits, dim=1)
        max_known = torch.sum(soft_logits[:, :54], dim=1)[0]
        unknown = soft_logits[:, 54]
        margin = torch.abs(max_known - unknown) - 0.5
        return -torch.min(margin, torch.zeros_like(margin)).mean()
    
    @torch.cuda.amp.autocast()
    def _compute_style_loss(self, 
                       image_features: torch.Tensor, 
                       dom_label: torch.Tensor, 
                       label: torch.Tensor, 
                       attri: torch.Tensor, 
                       mask_embed: torch.Tensor, 
                       logit_scale: torch.Tensor) -> tuple:
        device = image_features.device
        sty_embedding_list = []
        logits_list = []
        labels_list = []
        t=0
        loss_sty=0
        for domain in [0, 1, 2]:
            domain_mask = dom_label == domain
            domain_features = image_features[domain_mask]
            original_indices = torch.nonzero(domain_mask, as_tuple=False).squeeze(1)
            domain_labels = label[domain_mask]
            
            # Skip if no features for this domain
            if domain_features.size(0) == 0:
                # print("Yes")
                continue
            # else:
            #     print("No")
                
            # Generate prompts for current domain
            t+=1
            d = self.domainnames[domain]
            tokenized_prompts = torch.cat([
                clip.tokenize(f"A {d.replace('_', ' ')} of a {p}") 
                for p in self.classnames[:-1]
            ]).to(device)
            # with open('/raid/biplab/hassan/ODG-CLIP-LORA/attribute_domainnet.txt', 'r') as file:
            #     lines = file.readlines()
            # # # tokenized_prompts = torch.cat([clip.tokenize(f"a {d} of {",".join(a.split(",")[:4])} {p}") for a,p in zip(lines,self.classnames[:-1])]+[clip.tokenize(f"a {d} of {self.classnames[-1]}")]).cuda()
            # tokenized_prompts = torch.cat(
            #         [clip.tokenize(f"a {d} of {','.join(a.split(',')[:4])} {p}") for a, p in zip(lines, self.classnames[:-1])]
            #     ).cuda()
            n_cls = len(self.classnames)
            # # Compute attention
            domain_img = einops.repeat(domain_features, 'm n -> k m n', k=n_cls-1)
            cross_atten = self.cross_attention(
                domain_img, 
                self.projector(attri), 
                mask_embed
            )
            
            # Get base embedding
            with torch.no_grad():
                embedding = clip1.token_embedding(tokenized_prompts).type(self.dtype)
            
            # Process each feature in the domain
            domain_embeddings = []
            domain_logits = []
    
            for i in range(domain_features.size(0)):
                # with torch.set_grad_enabled(self.training):
                    embedding_copy = embedding.clone()
                    ctx_i = cross_atten[:,i,:].unsqueeze(1)
                    embedding_copy[:, 1:5, :] += ctx_i
                    # embedding_copy[:, 1:5, :] += self.projector1(domain_features[i]).unsqueeze(0).unsqueeze(0)
                    
                    # Compute embeddings and normalize
                    # with torch.no_grad():
                    embedding_int = self.text_encoder(embedding_copy, tokenized_prompts)
                    embedding_int = F.normalize(embedding_int, dim=-1)
                    
                    # Compute logits
                    logit = logit_scale * domain_features[i] @ embedding_int.t()
                    
                    domain_logits.append(logit)
                    domain_embeddings.append(embedding_int)
            
            # Stack results
            domain_logits = torch.stack(domain_logits)
            domain_embeddings = torch.stack(domain_embeddings)
            
            # Store results with original indices
            for i, idx in enumerate(original_indices):
                sty_embedding_list.append((idx.item(), domain_embeddings[i]))

            loss_sty += F.cross_entropy(domain_logits, domain_labels)
            
            # logits_list.append(domain_logits)
            # labels_list.append(domain_labels)
        
        # Combine and compute final loss
        # if not logits_list or not labels_list:
        #     raise ValueError("No valid domains found in the batch")
            
        # combined_logits = torch.cat(logits_list)
        # combined_labels = torch.cat(labels_list)
        
        # Sort embeddings by original index
        # print(combined_/
        loss_sty/=t
        sty_embedding_list.sort(key=lambda x: x[0])
        sty_embedding = torch.cat([x[1] for x in sty_embedding_list])
        # print(sty_embedding.shape)
        
        # loss_sty = F.cross_entropy(combined_logits, combined_labels)
        
        return loss_sty, sty_embedding
    