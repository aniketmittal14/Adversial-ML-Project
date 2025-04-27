import torch
import numpy as np
import torch.nn.functional as F

class DeformablePatchRepresentation:
    def __init__(self, height, width, center_rel, num_rays, dpr_lambda=-100.0, device='cuda'):
        """
        Initializes the DPR module.
        Args:
            height (int): Height of the image/mask.
            width (int): Width of the image/mask.
            center_rel (list or tuple): Relative center [x, y] (0.0 to 1.0).
            num_rays (int): Number of rays R.
            dpr_lambda (float): Sharpness parameter for activation.
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.H = height
        self.W = width
        self.center_x = center_rel[0] * (width - 1)
        self.center_y = center_rel[1] * (height - 1)
        self.num_rays = num_rays
        self.dpr_lambda = dpr_lambda
        self.device = device

        self.delta_theta = 2 * np.pi / num_rays

        # Precompute pixel coordinates and angles relative to center
        self.pixel_coords = self._precompute_pixel_coords() # Shape (H*W, 2)
        self.pixel_angles = self._precompute_pixel_angles() # Shape (H*W,)
        self.pixel_distances = self._precompute_pixel_distances() # Shape (H*W,)
        self.pixel_ray_indices = self._assign_pixels_to_rays() # Shape (H*W,)

    def _precompute_pixel_coords(self):
        """Generates grid of pixel coordinates."""
        y_coords, x_coords = torch.meshgrid(torch.arange(self.H, device=self.device),
                                            torch.arange(self.W, device=self.device),
                                            indexing='ij')
        coords = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1) # (H*W, 2) [x, y]
        return coords

    def _precompute_pixel_angles(self):
        """Calculates the angle of each pixel relative to the center."""
        # Shift coordinates so center is origin
        relative_coords = self.pixel_coords - torch.tensor([[self.center_x, self.center_y]], device=self.device)
        # Calculate angles (atan2 handles quadrants correctly)
        angles = torch.atan2(relative_coords[:, 1], relative_coords[:, 0]) # range [-pi, pi]
        # Shift to range [0, 2*pi]
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        return angles

    def _precompute_pixel_distances(self):
        """Calculates the distance of each pixel from the center."""
        relative_coords = self.pixel_coords - torch.tensor([[self.center_x, self.center_y]], device=self.device)
        distances = torch.norm(relative_coords, p=2, dim=1)
        return distances

    def _assign_pixels_to_rays(self):
        """Assigns each pixel to the angular interval of a ray pair."""
        # Find which angular segment each pixel falls into
        # Segment i is between angle i*delta_theta and (i+1)*delta_theta
        indices = torch.floor(self.pixel_angles / self.delta_theta).long()
        # Handle edge case where angle is exactly 2*pi
        indices = torch.clamp(indices, 0, self.num_rays - 1)
        return indices

    def _get_ray_endpoints(self, ray_lengths_param):
        """Calculates the endpoints of rays based on current lengths."""
        # ray_lengths_param: tensor of shape (num_rays,)
        # Convert relative lengths to absolute pixel coordinates
        # Assumes max possible length corresponds to image diagonal for scaling, adjust if needed
        max_len_pixel = np.sqrt(self.H**2 + self.W**2) / 2 # Example scaling
        ray_lengths_pixel = ray_lengths_param * max_len_pixel

        angles = torch.arange(self.num_rays, device=self.device) * self.delta_theta
        end_x = self.center_x + ray_lengths_pixel * torch.cos(angles)
        end_y = self.center_y + ray_lengths_pixel * torch.sin(angles)
        endpoints = torch.stack((end_x, end_y), dim=1) # Shape: (num_rays, 2) [x, y]
        return endpoints

    def _activation(self, x):
        """Differentiable approximation of step function."""
        # Phi(x) = (tanh(lambda * (x - 1)) + 1) / 2
        # Input x corresponds to |CO| / |DO| ratio
        return (torch.tanh(self.dpr_lambda * (x - 1)) + 1) / 2

    def forward(self, ray_lengths_param):
        """
        Generates the differentiable mask.
        Args:
            ray_lengths_param (torch.Tensor): Optimizable tensor of ray lengths (relative), shape (num_rays,).
        Returns:
            torch.Tensor: Differentiable mask, shape (H, W).
        """
        if not ray_lengths_param.requires_grad:
             print("Warning: ray_lengths_param does not require grad in DPR forward.")

        # 1. Get ray endpoints P based on current lengths
        ray_endpoints = self._get_ray_endpoints(ray_lengths_param) # (num_rays, 2)

        # 2. For each pixel C, determine its corresponding ray pair (A, B)
        # We use the precomputed pixel_ray_indices
        # Point A corresponds to ray index i, Point B to ray index (i+1) % num_rays
        indices_A = self.pixel_ray_indices
        indices_B = (self.pixel_ray_indices + 1) % self.num_rays

        point_A = ray_endpoints[indices_A] # (H*W, 2)
        point_B = ray_endpoints[indices_B] # (H*W, 2)
        point_O = torch.tensor([[self.center_x, self.center_y]], device=self.device) # (1, 2)
        point_C = self.pixel_coords # (H*W, 2)

        # 3. Calculate intersection point D of line CO and line AB
        # Vector AB = B - A
        vec_AB = point_B - point_A
        # Vector OC = C - O
        vec_OC = point_C - point_O

        # Using line equations and solving for intersection is prone to numerical issues
        # Alternative approach: Use cross products / areas if geometry is simpler
        # Here, implementing the intersection approach from paper (requires solving linear system)
        # Line AB: P = A + t * (B - A) => (x, y) = (Ax + t*ABx, Ay + t*ABy)
        # Line CO: P = O + u * (C - O) => (x, y) = (Ox + u*OCx, Oy + u*OCy)
        # We need to find the intersection D, which lies on AB.
        # Specifically, we need the length |OD| where D is intersection of extended CO with AB.
        # Let C = (Cx, Cy), O = (Ox, Oy), A = (Ax, Ay), B = (Bx, By)
        # Line AB: (By-Ay)x - (Bx-Ax)y = Ax(By-Ay) - Ay(Bx-Ax)
        # Line CO: (Cy-Oy)x - (Cx-Ox)y = Ox(Cy-Oy) - Oy(Cx-Ox)

        # Using a more robust method: find parameter t for D on AB such that O, C, D are collinear.
        # D = A + t*(B-A). Vector OD = D - O. Vector OC = C - O.
        # OD and OC must be parallel, so cross product (in 2D) is zero:
        # ODx * OCy - ODy * OCx = 0
        # (Ax + t*ABx - Ox) * OCy - (Ay + t*ABy - Oy) * OCx = 0
        # Solve for t:
        # t * (ABx*OCy - ABy*OCx) = (Ox-Ax)*OCy - (Oy-Ay)*OCx
        denominator = vec_AB[:, 0] * vec_OC[:, 1] - vec_AB[:, 1] * vec_OC[:, 0]
        numerator = (point_O[0, 0] - point_A[:, 0]) * vec_OC[:, 1] - \
                    (point_O[0, 1] - point_A[:, 1]) * vec_OC[:, 0]

        # Add epsilon to avoid division by zero
        epsilon = 1e-8
        # Handle cases where AB and OC are parallel (denominator is zero) - pixel lies on ray boundary?
        # If denominator is near zero, the lines might be parallel or C might be O.
        # Assign a default behavior (e.g., ratio = large value -> mask=0) ? Needs careful check.
        t = numerator / (denominator + torch.sign(denominator)*epsilon + epsilon) # Add signed epsilon

        # Calculate intersection point D
        point_D = point_A + t.unsqueeze(1) * vec_AB # (H*W, 2)

        # 4. Calculate distances |CO| and |DO|
        # |CO| is precomputed as self.pixel_distances
        dist_CO = self.pixel_distances
        dist_DO = torch.norm(point_D - point_O, p=2, dim=1)

        # 5. Calculate the ratio and pass through activation
        # Avoid division by zero for dist_DO
        ratio = dist_CO / (dist_DO + epsilon)

        # If C is the center O, ratio is 0. If denominator was ~0, ratio might be large.
        # If the pixel C is outside the angular wedge AOB, this calculation might be invalid.
        # DPR paper assumes C is within the angular wedge. Our precomputation assigns it.

        # Handle cases where D is between O and C (ratio > 1), or O is between C and D (ratio < 0?)
        # The activation function handles ratio > 1 correctly -> output near 0.
        # If t is outside [0, 1], D is outside segment AB. This can happen if C is far out. Ratio handles this.
        # If dist_DO is very small (D is near O), ratio becomes large -> output 0.

        mask_values_flat = self._activation(ratio)

        # Pixels exactly at the center: ratio=0, activation=0.5? Set to 1?
        # The paper sets M_O=1 (Algorithm 1, line 13).
        center_pixel_index = int(self.center_y * self.W + self.center_x)
        mask_values_flat[center_pixel_index] = 1.0 # Ensure center is always included

        # 6. Reshape flat mask to (H, W)
        mask = mask_values_flat.view(self.H, self.W)

        return mask # Shape (H, W)