import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import FeatureExtractorWrapper # Assuming models.py defines this

def gram_matrix(features):
    """ Calculates the Gram matrix for style loss. """
    # Input shape: (B, C, H, W)
    b, c, h, w = features.size()
    features_flat = features.view(b * c, h * w) # Or (b, c, h*w) -> (b, c, c) ? Check AdvCam/Gatys
    # Original Gatys paper: Reshape to (C, H*W), then compute F * F.T
    features_flat_gatys = features.view(c, h*w)
    gram = torch.mm(features_flat_gatys, features_flat_gatys.t())
    # Normalize by number of elements
    return gram.div(c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractorWrapper, style_image: torch.Tensor, device='cuda'):
        super().__init__()
        self.feature_extractor = feature_extractor.to(device)
        self.style_image = style_image.to(device)
        # Precompute target Gram matrices
        with torch.no_grad():
            style_features, _ = self.feature_extractor(self.style_image.unsqueeze(0))
            self.target_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    def forward(self, input_image):
        # input_image shape: (B, C, H, W) - ensure B=1 if not batching
        input_style_features, _ = self.feature_extractor(input_image)
        loss = 0.0
        for layer in self.target_grams:
            input_gram = gram_matrix(input_style_features[layer])
            loss += F.mse_loss(input_gram, self.target_grams[layer])
        return loss

class ContentLoss(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractorWrapper, content_image: torch.Tensor, device='cuda'):
        super().__init__()
        self.feature_extractor = feature_extractor.to(device)
        self.content_image = content_image.to(device)
        # Precompute target content features
        with torch.no_grad():
            _, self.target_features = self.feature_extractor(self.content_image.unsqueeze(0))

    def forward(self, input_image):
        _, input_content_features = self.feature_extractor(input_image)
        loss = 0.0
        for layer in self.target_features:
            loss += F.mse_loss(input_content_features[layer], self.target_features[layer])
        return loss

class SmoothnessLoss(nn.Module):
    """ Total Variation Loss """
    def forward(self, image):
        # image shape: (B, C, H, W)
        loss_x = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
        loss_y = torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        return loss_x + loss_y

class AdversarialLoss(nn.Module):
    def __init__(self, attack_type='untargeted', target_class=None):
        super().__init__()
        self.attack_type = attack_type
        self.target_class = target_class
        if attack_type == 'targeted' and target_class is None:
            raise ValueError("Target class must be specified for targeted attack.")
        # Use CrossEntropyLoss for standard classification loss calculation
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, model_outputs, true_labels):
        """
        Args:
            model_outputs (torch.Tensor): Logits from the target model. Shape (B, NumClasses)
            true_labels (torch.Tensor): Original true labels. Shape (B,)
        """
        batch_size = model_outputs.shape[0]

        if self.attack_type == 'targeted':
            # We want to maximize probability of target_class
            # Equivalent to minimizing CE loss towards target_class
            target_labels = torch.full_like(true_labels, self.target_class)
            loss = self.criterion(model_outputs, target_labels)
            # AdvCam paper adds log(py(x')) term - check if needed or just use CE
        else: # Untargeted
            # We want to maximize probability of *any* wrong class
            # Equivalent to minimizing CE loss towards true_label, then *negating* it
            loss = -self.criterion(model_outputs, true_labels)

        return loss

class ShapeLoss(nn.Module):
    """ Penalizes area exceeding threshold """
    def __init__(self, area_threshold_rel, image_area):
        super().__init__()
        self.threshold_pixels = area_threshold_rel * image_area

    def forward(self, mask):
        """
        Args:
            mask (torch.Tensor): Differentiable mask (H, W), values approx [0, 1].
        """
        current_area = torch.sum(mask)
        # Penalize only if area exceeds threshold
        loss = F.relu(current_area - self.threshold_pixels)
        return loss