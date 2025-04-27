# src/attack.py

import torch
import torch.optim as optim
import torch.nn.functional as F # Import functional for softmax
import torchvision.transforms as T
from tqdm import tqdm
import yaml
import os

# ... (imports for other modules: dpr, losses, models, utils, transformations)
from .dpr import DeformablePatchRepresentation
from .losses import StyleLoss, ContentLoss, SmoothnessLoss, AdversarialLoss, ShapeLoss
from .models import get_target_model, get_feature_extractor, FeatureExtractorWrapper
from .utils import load_image, save_image, apply_patch, get_patch_texture_param, get_ray_lengths_param, clip_ray_lengths
from .transformations import EOTTransform

# Function to load labels (as defined previously)
def load_imagenet_labels(filepath='imagenet_classes.txt'):
    try:
        with open(filepath, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except FileNotFoundError:
        print(f"Warning: Label file not found at {filepath}. Labels will not be displayed.")
        return None

class ShapeCamAttack:
    def __init__(self, config_path):
        # ... (Initialization code remains the same) ...
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = torch.device(self.config['device'])

        # --- Define Normalization ---
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

        # --- Load Labels ---
        self.imagenet_labels = load_imagenet_labels()

        # --- Load Models ---
        self.target_model = get_target_model(self.config['target_model_name']).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad_(False)
        self.target_model.eval()

        vgg_base = get_feature_extractor(self.config['feature_extractor_name']).to(self.device)
        style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        content_layers = ['relu4_2']
        self.feature_extractor = FeatureExtractorWrapper(vgg_base, style_layers, content_layers)
        self.feature_extractor.eval()

        # --- Load Data ---
        self.target_image = load_image(self.config['target_image_path']).to(self.device)
        self.style_image = load_image(self.config['style_image_path'], size=self.target_image.shape[1:]).to(self.device)
        self.C, self.H, self.W = self.target_image.shape
        with torch.no_grad():
             image_for_pred_norm = self.normalize(self.target_image.unsqueeze(0))
             outputs = self.target_model(image_for_pred_norm)
             self.true_label_idx = torch.argmax(outputs, dim=1).to(self.device)

        # --- Initialize Optimizable Parameters ---
        self.ray_lengths = get_ray_lengths_param(self.config['num_rays'], self.config['initial_ray_length'], self.device)
        self.patch_texture = get_patch_texture_param(self.H, self.W, self.C, self.device)

        # --- Setup DPR ---
        self.dpr = DeformablePatchRepresentation(self.H, self.W,
                                                 self.config['patch_center'],
                                                 self.config['num_rays'],
                                                 self.config['dpr_lambda'],
                                                 self.device)

        # --- Setup Losses ---
        self.l_style = StyleLoss(self.feature_extractor, self.style_image, self.device)
        self.l_content = ContentLoss(self.feature_extractor, self.target_image, self.device) if float(self.config['weight_content']) > 0 else None
        self.l_smooth = SmoothnessLoss() if float(self.config['weight_smoothness']) > 0 else None
        self.l_adv = AdversarialLoss(self.config['attack_type'], self.config.get('target_class'))
        self.l_shape = ShapeLoss(self.config['shape_area_threshold'], self.H * self.W)

        # --- Setup Optimizer ---
        params_to_optimize = [
            {'params': self.ray_lengths, 'lr': self.config['learning_rate_shape']},
            {'params': self.patch_texture, 'lr': self.config['learning_rate_texture']}
        ]
        if self.config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(params_to_optimize)
        else:
            self.optimizer = optim.SGD(params_to_optimize)

        # --- Optional EOT ---
        self.eot_transform = None
        if self.config.get('use_eot', False):
            self.eot_transform = EOTTransform(self.config['eot_rotation_range'],
                                              self.config['eot_scale_range'],
                                              self.config['eot_color_shift_std'],
                                              self.config['eot_samples'])
        self.mask_sharpened = False


    def attack(self):
        """Runs the main attack loop."""
        # ... (Setup output directories and log file as before) ...
        output_root = self.config['output_dir']
        log_dir = os.path.join(output_root, 'logs')
        adv_example_dir = os.path.join(output_root, 'adversarial_examples')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(adv_example_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'attack_log.txt')

        print("Starting attack...")
        for i in tqdm(range(self.config['max_iterations'])):
            self.optimizer.zero_grad()

            # --- Forward Pass: Generate Mask and Composite Image ---
            if self.mask_sharpened:
                with torch.no_grad():
                    current_mask = self.dpr.forward(self.ray_lengths)
                    current_mask = (current_mask > 0.5).float()
            else:
                 current_mask = self.dpr.forward(self.ray_lengths)

            current_patch_texture_clipped = torch.clamp(self.patch_texture, 0, 1)
            adv_image = apply_patch(self.target_image,
                                    current_patch_texture_clipped.squeeze(0),
                                    current_mask)
            adv_image_batch = adv_image.unsqueeze(0)

            # --- Calculate Losses (Using Normalized Images where needed) ---
            total_loss = 0
            loss_dict = {}

            adv_image_norm_batch = self.normalize(adv_image_batch) # Normalize for VGG losses

            # Style Loss
            l_style_val = self.l_style(adv_image_norm_batch)
            total_loss += float(self.config['weight_style']) * l_style_val
            loss_dict['style'] = l_style_val.item()

            # Content Loss
            if self.l_content:
                l_content_val = self.l_content(adv_image_norm_batch)
                total_loss += float(self.config['weight_content']) * l_content_val
                loss_dict['content'] = l_content_val.item()

            # Prepare image for Adversarial Loss (handle EOT, normalize)
            if self.eot_transform:
                 image_for_adv_batch_eot = self.eot_transform(adv_image_batch)
                 image_for_adv_norm = self.normalize(image_for_adv_batch_eot)
            else:
                 image_for_adv_norm = self.normalize(adv_image_batch)

            # Adversarial Loss
            outputs = self.target_model(image_for_adv_norm) # Get Logits
            l_adv_val = self.l_adv(outputs, self.true_label_idx)
            total_loss += float(self.config['weight_adv']) * l_adv_val
            loss_dict['adv'] = l_adv_val.item()

            # Smoothness Loss
            if self.l_smooth:
                l_smooth_val = self.l_smooth(self.patch_texture)
                total_loss += float(self.config['weight_smoothness']) * l_smooth_val
                loss_dict['smooth'] = l_smooth_val.item()

            # Shape Loss
            if not self.mask_sharpened:
                 l_shape_val = self.l_shape(current_mask)
                 total_loss += float(self.config['weight_shape']) * l_shape_val
                 loss_dict['shape'] = l_shape_val.item()

            # --- Backward Pass & Optimization ---
            total_loss.backward()
            if self.mask_sharpened and self.ray_lengths.grad is not None:
                 self.ray_lengths.grad.zero_()
            self.optimizer.step()

            # --- Clipping Parameters ---
            with torch.no_grad():
                self.ray_lengths.data = clip_ray_lengths(self.ray_lengths.data)

            # --- Logging & Saving Intermediate ---
            if i % 50 == 0 or i == self.config['max_iterations'] - 1:
                # Get prediction and confidence score
                with torch.no_grad(): # Ensure no grads calculation here
                    probabilities = F.softmax(outputs, dim=1) # Convert logits to probabilities
                    current_conf, current_pred_idx_tensor = torch.max(probabilities, dim=1) # Get max prob and its index
                    current_pred_idx = current_pred_idx_tensor.item()
                    current_conf_val = current_conf.item()

                true_idx = self.true_label_idx.item()
                pred_label = self.imagenet_labels[current_pred_idx] if self.imagenet_labels and 0 <= current_pred_idx < len(self.imagenet_labels) else "Unknown"
                true_label = self.imagenet_labels[true_idx] if self.imagenet_labels and 0 <= true_idx < len(self.imagenet_labels) else "Unknown"

                # --- Modify log string ---
                log_str = (f"Iter {i}: Loss: {total_loss.item():.4f} "
                           f"(Adv: {loss_dict.get('adv', 0):.3f}, "
                           f"Style: {loss_dict.get('style', 0):.3f}, "
                           f"Shape: {loss_dict.get('shape', 0):.3f}) | "
                           # --- ADDED Confidence Score ---
                           f"Pred: {pred_label} ({current_pred_idx}) @ {current_conf_val:.4f}, "
                           f"True: {true_label} ({true_idx})")
                # --- End log string modify ---

                print(log_str)
                with open(log_file, 'a') as f:
                    f.write(log_str + '\n')

                save_image(adv_image, os.path.join(adv_example_dir, f'adv_iter_{i}.png'))
                save_image(current_mask, os.path.join(adv_example_dir, f'mask_iter_{i}.png'))

            # --- Mask Sharpening Logic ---
            # ... (remains the same) ...
            sharpen_iter = self.config.get('sharpen_mask_at_iter', self.config['max_iterations'])
            if not self.mask_sharpened and i >= sharpen_iter:
                 print(f"--- Sharpening mask at iteration {i} ---")
                 self.mask_sharpened = True

        # ... (Attack finished message and final saving remain the same) ...
        print("Attack finished.")
        final_adv_image = adv_image
        final_mask = current_mask
        save_image(final_adv_image, os.path.join(adv_example_dir, 'final_adv.png'))
        save_image(final_mask, os.path.join(adv_example_dir, 'final_mask.png'))
        torch.save(self.ray_lengths.detach().cpu(), os.path.join(adv_example_dir, 'final_rays.pt'))
        torch.save(self.patch_texture.detach().cpu(), os.path.join(adv_example_dir, 'final_texture.pt'))

        return final_adv_image, final_mask