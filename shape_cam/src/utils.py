import torch
import torchvision.transforms as T
from torchvision.io import read_image, write_png
import numpy as np
import torch.nn.functional as F

def load_image(image_path, size=None):
    """Loads an image, normalizes to [0, 1], and optionally resizes."""
    img = read_image(image_path).float() / 255.0
    if size:
        img = T.Resize(size)(img)
    # Ensure 3 channels (handle grayscale)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] > 3: # RGBA?
        img = img[:3, :, :]
    return img # Shape: (C, H, W)

def save_image(tensor, file_path):
    """Saves a tensor image (range [0, 1]) to a file."""
    tensor = torch.clamp(tensor, 0, 1)
    # Ensure tensor is on CPU and detach from graph
    tensor_to_save = tensor.cpu().detach()

    # --- ADD THIS BLOCK ---
    # Handle 2D tensors (like masks) by adding a channel dimension
    if tensor_to_save.dim() == 2:
        tensor_to_save = tensor_to_save.unsqueeze(0) # Becomes (1, H, W)
    # --- END OF ADDED BLOCK ---

    # Handle batch dimension if present (should come after 2D check)
    elif tensor_to_save.dim() == 4:
        tensor_to_save = tensor_to_save[0] # Take the first image in the batch

    # Final check before saving
    if tensor_to_save.dim() != 3:
        raise ValueError(f"Tensor must be 3D (C, H, W) to save as PNG, but got shape {tensor_to_save.shape}")

    # Convert to byte and save
    write_png((tensor_to_save * 255).byte(), file_path)

def apply_patch(background_image, patch_texture, mask):
    """
    Applies the patch texture to the background image using the mask.
    Assumes background_image, patch_texture, mask are tensors on the same device.
    Mask should be broadcastable to image shape (e.g., 1, H, W or C, H, W).
    """
    # Ensure mask is broadcastable (e.g., (1, H, W) -> (C, H, W))
    if mask.dim() == 3 and mask.shape[0] == 1:
       mask = mask.repeat(background_image.shape[0], 1, 1)
    elif mask.dim() == 2: # (H, W) -> (1, H, W) -> (C, H, W)
        mask = mask.unsqueeze(0).repeat(background_image.shape[0], 1, 1)

    # Combine background and patch
    attacked_image = patch_texture * mask + background_image * (1 - mask)
    return torch.clamp(attacked_image, 0, 1)

def get_patch_texture_param(height, width, channels=3, device='cuda'):
    """Creates an optimizable tensor for the patch texture."""
    # Initialize randomly or with a solid color
    patch_param = torch.rand(1, channels, height, width, requires_grad=True, device=device)
    # Or: patch_param = torch.full((1, channels, height, width), 0.5, requires_grad=True, device=device)
    return patch_param

def get_ray_lengths_param(num_rays, initial_length, device='cuda'):
    """Creates an optimizable tensor for ray lengths."""
    # Initialize all rays to the initial length
    ray_lengths = torch.full((num_rays,), float(initial_length), requires_grad=True, device=device)
    return ray_lengths

def clip_ray_lengths(ray_lengths, min_val=0.001, max_val=0.7):
    """Clips ray lengths to be non-negative and within reasonable bounds (e.g., < sqrt(2)/2)."""
    # Max val prevents rays from going way outside the image for relative coords
    return torch.clamp(ray_lengths, min=min_val, max=max_val)

# --- Add Normalization/Denormalization if needed for specific models ---
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)
# denormalize = T.Normalize(mean=[-m/s for m, s in zip(imagenet_mean, imagenet_std)],
#                          std=[1/s for s in imagenet_std])