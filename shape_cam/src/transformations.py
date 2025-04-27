import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np

class EOTTransform:
    """Applies random transformations for Expectation Over Transformation."""
    def __init__(self, rotation_range, scale_range, color_shift_std, n_samples=1):
        self.rotation_degrees = rotation_range
        self.scale_factors = scale_range
        self.color_shift_std = color_shift_std
        self.n_samples = n_samples
        # TODO: Add random background pasting if needed

    def __call__(self, image_batch):
        """Applies random transformations to a batch of images."""
        # Assumes image_batch is (B, C, H, W)
        transformed_batches = []
        for _ in range(self.n_samples):
            b, c, h, w = image_batch.shape
            transformed_images = []
            for i in range(b):
                img = image_batch[i]
                # 1. Rotation
                angle = random.uniform(self.rotation_degrees[0], self.rotation_degrees[1])
                img = TF.rotate(img, angle, interpolation=T.InterpolationMode.BILINEAR)

                # 2. Scaling (Resized Crop)
                scale = random.uniform(self.scale_factors[0], self.scale_factors[1])
                new_h, new_w = int(h * scale), int(w * scale)
                img = TF.resize(img, (new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)
                # Crop back to original size (random or center crop)
                img = T.CenterCrop((h, w))(img) # Or RandomCrop

                # 3. Color Jitter (simplified)
                jitter = torch.randn(c, 1, 1, device=img.device) * self.color_shift_std
                img = img + jitter
                img = torch.clamp(img, 0, 1)

                transformed_images.append(img)

            transformed_batches.append(torch.stack(transformed_images))

        # Average over samples or return all? AdvCam likely averages loss over samples
        # For simplicity here, just return the first sample batch
        if self.n_samples > 0:
             return transformed_batches[0] # Or handle multiple samples in loss calculation
        else:
             return image_batch