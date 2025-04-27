# main.py
import argparse
import os
import torch
import numpy as np
import random # Import random module
from src.attack import ShapeCamAttack

# --- SEED SETTING ---
SEED = 14 # Or any integer you choose
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # if you are using multi-GPU
# --- END SEED SETTING ---


def main():
    parser = argparse.ArgumentParser(description='Shape-Optimized Adversarial Camouflage Attack')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the configuration file.')
    # Add overrides for config parameters if needed, e.g.:
    # parser.add_argument('--target_image', type=str, help='Override target image path')
    # parser.add_argument('--style_image', type=str, help='Override style image path')
    # parser.add_argument('--output_dir', type=str, help='Override output directory')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    # --- Initialize and run the attack ---
    attack_runner = ShapeCamAttack(args.config)
    # TODO: Potentially override config values here based on args before running attack
    # if args.target_image: attack_runner.config['target_image_path'] = args.target_image
    # ...

    final_adv_image, final_mask = attack_runner.attack()

    print(f"Attack complete. Results saved in {attack_runner.config['output_dir']}")

if __name__ == '__main__':
    main()