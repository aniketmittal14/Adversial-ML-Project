import torch
import torchvision.models as models
import torch.nn as nn

def get_target_model(model_name, pretrained=True):
    """Loads a pre-trained target classifier."""
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    # Add more models as needed
    else:
        raise ValueError(f"Model {model_name} not supported.")
    model.eval() # Set to evaluation mode
    return model

def get_feature_extractor(model_name, requires_grad=False):
    """Loads a pre-trained model (like VGG) to extract features for style/content loss."""
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=True).features
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True).features
    else:
        raise ValueError(f"Feature extractor {model_name} not supported.")

    if not requires_grad:
        for param in model.parameters():
            param.requires_grad_(False)
    model.eval()
    return model

class FeatureExtractorWrapper(nn.Module):
    """ A wrapper to easily extract features from specified layers of a base model. """
    def __init__(self, base_model, style_layers, content_layers):
        super().__init__()
        self.base_model = base_model
        # These will now be dictionaries {name: index}
        self.style_layer_indices = self._get_layer_indices(style_layers)
        self.content_layer_indices = self._get_layer_indices(content_layers)

        # CHANGE: Calculate max index from dictionary values
        style_indices = list(self.style_layer_indices.values())
        content_indices = list(self.content_layer_indices.values())
        all_indices = style_indices + content_indices
        self.max_layer_index = max(all_indices) if all_indices else -1 # Handle case where no layers are found


    def _get_layer_indices(self, layer_names):
        # This needs adjustment based on the actual structure of base_model (e.g., VGG features)
        # Example for VGG features sequence:
        layer_map = {'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
                     'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
                     'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13, 'conv3_3': 14, 'relu3_3': 15, 'conv3_4': 16, 'relu3_4': 17, 'pool3': 18,
                     'conv4_1': 19, 'relu4_1': 20, 'conv4_2': 21, 'relu4_2': 22, 'conv4_3': 23, 'relu4_3': 24, 'conv4_4': 25, 'relu4_4': 26, 'pool4': 27,
                     'conv5_1': 28, 'relu5_1': 29, 'conv5_2': 30, 'relu5_2': 31, 'conv5_3': 32, 'relu5_3': 33, 'conv5_4': 34, 'relu5_4': 35, 'pool5': 36}
        indices_dict = {}
        for name in layer_names:
            if name in layer_map:
                # CHANGE: Add name:index pair to dictionary
                indices_dict[name] = layer_map[name]
            else:
                print(f"Warning: Layer '{name}' not found in VGG feature map.")
        # CHANGE: Return the dictionary
        return indices_dict


    def forward(self, x):
        style_features = {}
        content_features = {}
        current_features = x
        for i, layer in enumerate(self.base_model):
            current_features = layer(current_features)

            # CHANGE: Check if index 'i' is in the *values* of the dict
            if i in self.style_layer_indices.values():
                # Find the layer name(s) corresponding to this index i
                # (There might be multiple names mapping to same index, though unlikely in VGG)
                layer_names = [name for name, idx in self.style_layer_indices.items() if idx == i]
                for name in layer_names: # Handle potential multiple names for same index
                    style_features[name] = current_features

            # CHANGE: Check if index 'i' is in the *values* of the dict
            if i in self.content_layer_indices.values():
                layer_names = [name for name, idx in self.content_layer_indices.items() if idx == i]
                for name in layer_names:
                    content_features[name] = current_features

            if i >= self.max_layer_index:
                break # No need to go further
        return style_features, content_features

# Example Usage:
# vgg_features = get_feature_extractor('vgg19')
# style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'] # Example from Gatys et al.
# content_layers = ['relu4_2']
# feature_extractor = FeatureExtractorWrapper(vgg_features, style_layers, content_layers)
# style_feats, content_feats = feature_extractor(image_tensor)