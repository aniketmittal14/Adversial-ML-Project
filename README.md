# ShapeCam: Adversarial Patch Attack

## Overview
ShapeCam is a novel adversarial patch attack method that optimizes both the shape and visual style of patches to create stealthy and effective attacks against Deep Neural Networks (DNNs). Combining Deformable Patch Representation (DPR) and Adversarial Camouflage (AdvCam)-style transfer, ShapeCam produces patches that achieve high misclassification rates while remaining visually inconspicuous. The project evaluates attacks on 50 ImageNet classes using a ResNet-50 model, with results detailed in the accompanying paper.

## Prerequisites
To run ShapeCam, you’ll need:

- **Python**: 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`
- **Hardware**: GPU recommended (CUDA-compatible for PyTorch)
- **Input Files**:
  - Target images for the attack
  - Style images (e.g., leaves, fabrics, bricks)
  - Configuration file: `configs/default_config.yaml`
- **ImageNet Classes**: `imagenet_classes.txt` provides class labels (e.g., "Tiger Shark" as class 3, "Tench" as class 0)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aniketmittal14/Adversial-ML-Project
   cd Adversial-ML-Project
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include:
   - `torch>=1.9.0`
   - `torchvision>=0.10.0`
   - `numpy>=1.20.0`
   - `pyyaml>=5.4.0`
   - `opencv-python` (or Pillow)
   - `matplotlib` (optional, for visualization)

4. **Prepare Configuration**:
   Verify `configs/default_config.yaml` is configured with:
   - Paths to target and style images
   - Output directory
   - Model settings (e.g., ResNet-50)
   - Attack parameters (e.g., learning rates: 1e-2 for texture, 5e-3 for shape)

## Running the Code
Execute the ShapeCam attack with:

```bash
python main.py --config configs/default_config.yaml
```

### Details
- **`main.py`**: Entry point script that runs the `ShapeCamAttack` class.
- **`--config`**: Path to the configuration file (`configs/default_config.yaml` by default).
- **Output**: Generates an adversarial image and mask, saved in the config-specified output directory. A success message indicates the save location.

### Example
With `configs/default_config.yaml` set for a "Tiger Shark" target image and a leaf texture style, the attack produces a patch that misclassifies the image as "Tench" with high stealthiness.

## Project Structure
- `main.py`: Runs the attack.
- `src/attack.py`: Implements the `ShapeCamAttack` class.
- `configs/default_config.yaml`: Configures attack parameters.
- `requirements.txt`: Lists dependencies.
- `imagenet_classes.txt`: Defines ImageNet class labels.
- `README.md`: This documentation.

## Notes
- **Customization**: Add arguments like `--target_image` to `main.py` for config overrides (requires script modification).
- **Dependencies**: Use Pillow instead of OpenCV if preferred, updating `src/attack.py` accordingly.
- **Troubleshooting**: If `configs/default_config.yaml` is missing, the script will error. Check the file’s path and contents.
- **Paper Reference**: See the LaTeX document for methodology and results (e.g., 90.5% attack success rate for 3% patch area, "Tiger Shark" to "Tench" attack).
