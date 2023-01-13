import argparse
import torch
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn
from typing import List
import math
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description= "Encode images")
parser.add_argument("--caption", type=str, help="image caption")
parser.add_argument("--vae", type=str, help="folder vae is located", default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--device", type=str, help="Device to use", default="cuda", required=False)
parser.add_argument("--unet", type=str, help="folder unet subfolder is located", default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--LGP_path", type=str, help="folder pre-trained LGP is located")
parser.add_argument("--noise_strength", type=float, help="denoising strength")
parser.add_argument("--image_path", type=str, help="folder skecth is located")

args = parser.parse_args()   
device = args.device
lgp_path = args.LGP_path
img_path = args.image_path

blocks = [0,1,2,3]
caption = args.caption
num_inference_steps = 50
batch_size = 1
guidance_scale = 8
strength = args.noise_strength
eta = 0.0

class latent_guidance_predictor(nn.Module):
    def __init__(self, output_dim, input_dim, num_encodings):
        super(latent_guidance_predictor, self).__init__()
        self.num_encodings = num_encodings
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),         
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),     
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),      
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, t):
        # Concatenate input pixels with noise level t and positional encodings
        pos_encoding = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_encodings)]
        pos_encoding = torch.cat(pos_encoding, dim=-1)
        x = torch.cat((x, t, pos_encoding), dim=-1)
        x = x.flatten(start_dim=0, end_dim=2)
        
        return self.layers(x)
    
model = latent_guidance_predictor(output_dim=4, input_dim=7080, num_encodings=9).to(device)
checkpoint = torch.load(lgp_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

vae = AutoencoderKL.from_pretrained(args.vae, subfolder= "vae", use_auth_token=False).to(device)
unet = UNet2DConditionModel.from_pretrained(args.unet, subfolder="unet", use_auth_token=False).to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

@torch.no_grad()
def to_latents(img:Image):
  np_img = (np.array(img).astype(np.float32) / 255.0) * 2.0 - 1.0
  np_img = np_img[None].transpose(0, 3, 1, 2)
  torch_img = torch.from_numpy(np_img)
  generator = torch.Generator(device).manual_seed(0)
  latents = vae.encode(torch_img.to(vae.dtype).to(device)).latent_dist.sample(generator=generator)
  latents = latents * 0.18215
  return latents

@torch.no_grad()
def to_img(latents):
  torch_img = vae.decode(latents.to(vae.dtype).to(device)).sample
  torch_img = (torch_img / 2 + 0.5).clamp(0, 1)
  np_img = torch_img.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
  np_img = (np_img * 255.0).astype(np.uint8)
  img = Image.fromarray(np_img)
  return img

def noisy_latent(image, noise_scheduler, timesteps):
  noise = torch.randn(image.shape).to(device)
  noisy_image = noise_scheduler.add_noise(image, noise, timesteps)
  sqrt_alpha_prod = noise_scheduler.alphas_cumprod[timesteps].to(device) ** 0.5
  sqrt_alpha_prod = sqrt_alpha_prod.flatten()
  while len(sqrt_alpha_prod.shape) < len(image.shape):
    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
  noise_level = noisy_image - (sqrt_alpha_prod * image)
  return noisy_image, noise_level

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() for f in features if f is not None and isinstance(f, torch.Tensor)]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())

def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out

def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

offset = noise_scheduler.config.get("steps_offset", 0)

noise_scheduler.set_timesteps(num_inference_steps)

# get the original timestep using init_timestep
init_timestep = int(num_inference_steps * strength) + offset
init_timestep = min(init_timestep, num_inference_steps)

if isinstance(noise_scheduler, LMSDiscreteScheduler):
    timesteps = torch.tensor([num_inference_steps - init_timestep] * batch_size, dtype=torch.long, device=device)
else:
    timesteps = noise_scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=device)
    
text_input = tokenizer([caption], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
  text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
[""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
with torch.no_grad():
  uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]   
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

def extract_features(latent_image, blocks):
    latent_model_input = torch.cat([latent_image] * 2)
    activations = []
    save_hook = save_out_hook
    feature_blocks = []
    for idx, block in enumerate(unet.down_blocks):
        if idx in blocks:
            block.register_forward_hook(save_hook)
            feature_blocks.append(block) 
            
    for idx, block in enumerate(unet.up_blocks):
        if idx in blocks:
            block.register_forward_hook(save_hook)
            feature_blocks.append(block)  
    with torch.no_grad():
        noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample

    # Extract activations
    for block in feature_blocks:
        activations.append(block.activations)
        block.activations = None
        
    activations = [activations[0][0], activations[1][0], activations[2][0], activations[3][0], activations[4], activations[5], activations[6], activations[7]]
    
    return activations

def resize_and_concatenate(activations: List[torch.Tensor], reference):
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = reference.shape[2:]
    resized_activations = []
    for acts in activations:
        acts = nn.functional.interpolate(
            acts, size=size, mode="bilinear"
        )
        acts = acts[:1]
        acts = acts.transpose(1,3)
        resized_activations.append(acts)
    
    return torch.cat(resized_activations, dim=3)

img = Image.open(img_path)
img = img.resize((512,512))
imagelatent = to_latents(img)

noisy_image, noise_level = noisy_latent(imagelatent, noise_scheduler, timesteps)
noise_level = noise_level.transpose(1,3)

file_name = os.path.basename(img_path)
img_name = os.path.splitext(file_name)[0]

features = extract_features(noisy_image, blocks)
features = resize_and_concatenate(features, imagelatent)

pred_edge_map = model(features, noise_level).unflatten(0, (1, 64, 64)).transpose(3, 1)
pred_edge_map = to_img(pred_edge_map)
pred_edge_map.save(img_name + '-edge_map.jpg')

print("Done!")