import os
import torch
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn
from typing import List
import math
import torch.nn as nn
from tqdm import tqdm
import inspect

parser = argparse.ArgumentParser(description= "Training the LGP")
parser.add_argument("--dataset_dir", type=str, help="folder training image dataset is located")
parser.add_argument("--edge_maps_dir", type=str, help="folder training image dataset is located")
parser.add_argument("--vae", type=str, help="folder vae subfolder is located", default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--unet", type=str, help="folder unet subfolder is located", default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--device", type=str, help="Device to use, defaults to gpu", default="cuda")
parser.add_argument("--epochs", type=int, help="number of training epochs", default=16)
parser.add_argument("--batch_size", type=int, help="batch size of dataloader", default=15)
parser.add_argument("--LGP_path", type=str, help="folder the trained LGP should be saved to", default="/workspace/trained_LGP")
parser.add_argument("--lr", type=float, help="Learning rate of the optimizer", default = 0.0001)

args = parser.parse_args()   
dataset_dir = args.dataset_dir
edge_maps_dir = args.edge_maps_dir
device = args.device

vae = AutoencoderKL.from_pretrained(args.vae, subfolder= "vae", use_auth_token=False).to(device)
unet = UNet2DConditionModel.from_pretrained(args.unet, subfolder="unet", use_auth_token=False).to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

noise_scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# Important Parameters
blocks = [0,1,2,3]
batch_size = 1
guidance_scale = 8
num_inference_steps = 50
eta = 0.0

offset = noise_scheduler.config.get("steps_offset", 0)

noise_scheduler.set_timesteps(num_inference_steps)

accepts_eta = "eta" in set(inspect.signature(noise_scheduler.step).parameters.keys())
extra_step_kwargs = {}
if accepts_eta:
    extra_step_kwargs["eta"] = eta

@torch.no_grad()
def img_to_latents(img:Image):
  np_img = (np.array(img).astype(np.float32) / 255.0) * 2.0 - 1.0
  np_img = np_img[None].transpose(0, 3, 1, 2)
  torch_img = torch.from_numpy(np_img)
  generator = torch.Generator(device).manual_seed(0)
  latents = vae.encode(torch_img.to(vae.dtype).to(device)).latent_dist.sample(generator=generator)
  latents = latents * 0.18215
  return latents

def noisy_latent(image, noise_scheduler):
  timesteps = torch.randint(250, 900, (1,), device=device).long()
  noise = torch.randn(image.shape).to(device)
  noisy_image = noise_scheduler.add_noise(image, noise, timesteps)
  sqrt_alpha_prod = noise_scheduler.alphas_cumprod[timesteps].to(device) ** 0.5
  sqrt_alpha_prod = sqrt_alpha_prod.flatten()
  while len(sqrt_alpha_prod.shape) < len(image.shape):
    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
  noise_level = noisy_image - (sqrt_alpha_prod * image)
  return noisy_image, noise_level, timesteps

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

def extract_features(latent_image, blocks, text_embeddings, timesteps):
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

class LGPDataset(Dataset):
    def __init__(self, dataset_dir, edge_maps_dir):
        self.dataset_dir = dataset_dir
        self.edge_maps_dir = edge_maps_dir
        self.image_paths = []
        
        # Iterate through all subfolders in dataset_dir
        for subfolder in os.listdir(self.dataset_dir):
            subfolder_path = os.path.join(self.dataset_dir, subfolder)
            caption = subfolder

            # Iterate through all images in the subfolder
            for image_file in os.listdir(subfolder_path):
                if image_file.endswith('.jpg'):
                    image_path = os.path.join(subfolder_path, image_file)
                    image_unsure = Image.open(image_path)
                    if image_unsure.mode != "RGB":
                        continue
                        
                    edge_map_file = image_file.replace('.jpg', '.png')
                    edge_map_path_unsure = os.path.join(self.edge_maps_dir, edge_map_file)
                    if os.path.exists(edge_map_path_unsure):
                        edge_map_path = edge_map_path_unsure
                        self.image_paths.append((image_path, edge_map_path, caption))
                    else:
                        continue       
        
    def __getitem__(self, index):
        image_path, edge_map_path, caption = self.image_paths[index]
        image = Image.open(image_path)
        image = image.resize((512,512))
        edge_map = Image.open(edge_map_path)
        rgb_edge_map = Image.merge('RGB', (edge_map, edge_map, edge_map))
        rgb_edge_map = rgb_edge_map.resize((512,512))
        encoded_edge_map = img_to_latents(rgb_edge_map)
        encoded_edge_map = encoded_edge_map.transpose(1,3)
        
        encoded_image = img_to_latents(image).to(device)
        noisy_image, noise_level, timesteps = noisy_latent(encoded_image, noise_scheduler)
        text_input = tokenizer([caption], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]   
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        features = extract_features(noisy_image, blocks, text_embeddings, timesteps)
        features =  resize_and_concatenate(features, encoded_image)
        noise_level = noise_level.transpose(1,3)
        
        return features, encoded_edge_map, noise_level

    def __len__(self):
        return len(self.image_paths)
    
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

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x, t):
        # Concatenate input pixels with noise level t and positional encodings
        pos_encoding = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_encodings)]
        pos_encoding = torch.cat(pos_encoding, dim=4)
        x = torch.cat((x, t, pos_encoding), dim=4)
        x = x.flatten(start_dim=0, end_dim=3)
        
        return self.layers(x)

dataset = LGPDataset(dataset_dir = dataset_dir, edge_maps_dir = edge_maps_dir)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

num_epochs = args.epochs
num_steps = len(dataloader) * args.epochs
LGP = latent_guidance_predictor(output_dim=4, input_dim=7080, num_encodings=9).to(device)
LGP.init_weights()
LGP.train()
optimizer = torch.optim.Adam(LGP.parameters(), lr=args.lr)
criterion = nn.MSELoss()

batch_count = 0
training_loader_iter = iter(dataloader)

def next_batch():
    features, encoded_edge_map, noise_level = next(training_loader_iter)
    features, encoded_edge_map, noise_level = features.to(device), encoded_edge_map.to(device), noise_level.to(device)
    encoded_edge_map = encoded_edge_map.flatten(start_dim=0, end_dim=3)
    
    return features, encoded_edge_map, noise_level

features, encoded_edge_map, noise_level = next_batch()

for step in tqdm(range(1, num_steps+1), desc="Step", position=0, leave=True):
    optimizer.zero_grad()
    output = LGP(features, noise_level)
    loss = criterion(output, encoded_edge_map)
    loss.backward()
    print(loss.item())
    optimizer.step()

    batch_count += 1
        
    if batch_count % args.epochs == 0 and batch_count != num_steps:
        features, encoded_edge_map, noise_level = next_batch()
                          
model_path = args.LGP_path + '.pt'
print('saving to:', model_path)
torch.save({
            'model_state_dict': LGP.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path)
