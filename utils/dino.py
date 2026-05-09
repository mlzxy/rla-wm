import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from enum import Enum
from typing import Tuple, Union
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_sample_image

    SKLEARN_AVAILABLE = True
except (ImportError, ValueError, RuntimeError):
    SKLEARN_AVAILABLE = False
    print(
        "Warning: Sklearn not available (binary incompatibility?), visualization disabled."
    )

# --- 1. Model Definitions ---


class DINOv3Model(Enum):
    """
    Enum for official DINOv3 ViT model identifiers on Hugging Face.
    These models are pretrained on the LVD-1689M dataset.
    """

    SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
    BASE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    LARGE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    HUGE = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    GIANT_7B = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


def get_dinov3_model_for_channels(vit_channels: int) -> DINOv3Model:
    """
    Select the appropriate DINOv3 model based on the number of channels.

    Args:
        vit_channels: Number of channels/embedding dimension.

    Returns:
        The appropriate DINOv3Model enum value.

    Typical embedding dimensions:
        - SMALL: 384
        - BASE: 768
        - LARGE: 1024
        - HUGE: 1280
        - GIANT_7B: 1536
    """
    if vit_channels <= 384:
        return DINOv3Model.SMALL
    elif vit_channels <= 768:
        return DINOv3Model.BASE
    elif vit_channels <= 1024:
        return DINOv3Model.LARGE
    elif vit_channels <= 1280:
        return DINOv3Model.HUGE
    else:
        return DINOv3Model.GIANT_7B


class DINOv3FeatureExtractor(nn.Module):
    """
    A wrapper class for extracting features using Meta's DINOv3 Vision Transformer models.

    This extractor always operates in evaluation mode with no gradients and uses float16
    for memory efficiency.
    """

    def __init__(
        self,
        model_name: Union[DINOv3Model, str] = DINOv3Model.SMALL,
        use_compile: bool = True,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        # Resolve model name from Enum if necessary
        self.model_name_str = (
            model_name.value if isinstance(model_name, DINOv3Model) else model_name
        )

        print(f"Loading DINOv3 model: {self.model_name_str}...")
        try:
            self.config = AutoConfig.from_pretrained(self.model_name_str)
            self.model = AutoModel.from_pretrained(
                self.model_name_str,
                config=self.config,
                attn_implementation=attn_implementation,
            )
        except (OSError, KeyError, ValueError) as e:
            # Fallback for testing if the specific DINOv3 repo is private/unavailable or model type unknown
            print(
                f"Warning: Could not load {self.model_name_str} (error: {e}). Loading DINOv2 for demo purposes."
            )
            self.model_name_str = "facebook/dinov2-small"
            self.config = AutoConfig.from_pretrained(self.model_name_str)
            self.model = AutoModel.from_pretrained(
                self.model_name_str, config=self.config
            )

        # Set to eval mode permanently
        self.model.eval()

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        if use_compile:
            print("Compiling DINOv3 model with torch.compile...")
            self.model = torch.compile(self.model)

        # Extract architectural details
        self.patch_size = getattr(self.config, "patch_size", 16)
        self.embed_dim = self.config.hidden_size

        # ImageNet normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "DINOv3FeatureExtractor":
        """Override train to always keep model in eval mode."""
        # Always keep in eval mode, ignore mode argument
        return super().train(False)

    def _preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # 1. Ensure float and normalize
        x = x.float()
        x = (x - self.mean) / self.std

        # 2. Pad to multiple of patch_size
        B, C, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        return x, H, W

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor, return_spatial_grid: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DINO features from input images.

        Args:
            x: Input tensor of shape [B, C, H, W], values in [0, 1] range
            return_spatial_grid: If True, reshape patch tokens to spatial grid [B, D, H', W']

        Returns:
            cls_token: CLS token features [B, D] (float16)
            patch_tokens: Patch features [B, D, H', W'] if return_spatial_grid else [B, N, D] (float16)
        """
        # Preprocess
        x_padded, H_orig, W_orig = self._preprocess(x)

        # Forward pass with automatic mixed precision (float16)
        device_type = "cuda" if x_padded.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            outputs = self.model(x_padded, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.last_hidden_state

        # Calculate grid dimensions
        B, N, D = last_hidden_state.shape
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        hH, hW = H_padded // self.patch_size, W_padded // self.patch_size
        num_patches = hH * hW
        num_extra_tokens = N - num_patches

        # Extract tokens
        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, num_extra_tokens:, :]

        if return_spatial_grid:
            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, D, hH, hW)

        return cls_token, patch_tokens

    @torch.inference_mode()
    def extract_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the CLS token from input images.

        Args:
            x: Input tensor of shape [B, C, H, W], values in [0, 1] range.

        Returns:
            cls_token: CLS token features [B, D] (float16).
        """
        cls_token, _ = self.forward(x, return_spatial_grid=False)
        return cls_token


# --- 2. Visualization Logic ---


def visualize_and_save(
    feature_map: torch.Tensor,
    original_img: torch.Tensor = None,
    output_path: str = "dinov3_features.png",
    n_components: int = 3,
):
    """
    Computes PCA on features, renders with Matplotlib, and saves via PIL.
    """
    # 1. Prepare Data (PCA Logic)
    # Take first image in batch: [D, H, W], convert to float32 for PCA
    feat = feature_map[0].detach().cpu().float()
    D, H, W = feat.shape

    # Flatten spatial dims: [N_pixels, D]
    feat_flat = feat.permute(1, 2, 0).reshape(-1, D)

    # PCA projection
    print(f"Running PCA to reduce {D} dims to {n_components}...")
    pca = PCA(n_components=n_components)
    feat_pca = pca.fit_transform(feat_flat.numpy())

    # Normalize PCA to 0-1 for visualization
    feat_min, feat_max = feat_pca.min(axis=0), feat_pca.max(axis=0)
    feat_pca = (feat_pca - feat_min) / (feat_max - feat_min)

    # Reshape back to spatial grid: [H, W, 3]
    pca_img = feat_pca.reshape(H, W, n_components)

    # 2. Setup Matplotlib Figure
    # 'Agg' backend prevents window popup (headless safe)
    plt.switch_backend("Agg")

    if original_img is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original Image
        img_np = original_img[0].permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image (Sklearn)")
        axes[0].axis("off")

        # PCA Features
        axes[1].imshow(pca_img)
        axes[1].set_title(f"DINO Features (PCA)")
        axes[1].axis("off")
    else:
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(pca_img)
        plt.title(f"DINO Features (PCA)")
        plt.axis("off")

    plt.tight_layout()

    # 3. Convert Matplotlib Figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)

    # Load image from buffer into PIL
    pil_image = Image.open(buf)

    # 4. Save using PIL
    print(f"Saving visualization to {output_path}...")
    pil_image.save(output_path)

    # Cleanup
    buf.close()
    plt.close(fig)


def visualize_dino_to_imgs(
    feature_map: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Converts DINO feature maps into RGB visualization images using PCA.

    Args:
        feature_map: Tensor of shape (B, C, dH, dW)
        patch_size: Resolution multiplier (default 16 for ViT-16)

    Returns:
        torch.Tensor: Visualization images of shape (B, 3, H, W), normalized to [0, 1].
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: SKLEARN not available. Returning zero images.")
        B, C, dH, dW = feature_map.shape
        return torch.zeros(
            (B, 3, dH * patch_size, dW * patch_size), device=feature_map.device
        )

    B, C, dH, dW = feature_map.shape
    # Flatten spatial dims and batch dim for PCA: (B * dH * dW, C)
    feat = feature_map.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().float().numpy()

    # PCA to 3 components
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat)  # (N, 3)

    # Normalize each component to [0, 1]
    f_min, f_max = feat_pca.min(axis=0), feat_pca.max(axis=0)
    # Avoid division by zero
    feat_pca = (feat_pca - f_min) / (f_max - f_min + 1e-8)

    # Reshape back to (B, dH, dW, 3) -> (B, 3, dH, dW)
    vis_map = torch.from_numpy(feat_pca).reshape(B, dH, dW, 3).permute(0, 3, 1, 2)

    # Upsample to full resolution
    vis_imgs = F.interpolate(
        vis_map,
        size=(dH * patch_size, dW * patch_size),
        mode="bilinear",
        align_corners=False,
    )

    return vis_imgs.to(feature_map.device).clamp(0, 1)


# --- 3. Main Execution ---

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A. Initial Load (Baseline - No Compile)
    extractor = DINOv3FeatureExtractor(
        model_name=DINOv3Model.SMALL, use_compile=False, attn_implementation="sdpa"
    )
    extractor = extractor.to(device)

    # B. Load Input Image
    if SKLEARN_AVAILABLE:
        print("Loading sample image from sklearn...")
        try:
            # 'china.jpg' or 'flower.jpg' are standard sklearn datasets
            raw_image_np = load_sample_image("china.jpg")  # Shape: (427, 640, 3)

            # Convert Numpy [H, W, C] -> Tensor [B, C, H, W]
            # Also normalize to [0, 1] range as expected by the extractor
            img_tensor = torch.from_numpy(raw_image_np).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dim
        except Exception as e:
            print(f"Warning: Failed to load sample image: {e}")
            SKLEARN_AVAILABLE = False
            img_tensor = torch.rand(1, 3, 427, 640)
    else:
        print("Loading sample image (random noise fallback)...")
        img_tensor = torch.rand(1, 3, 427, 640)

    img_tensor = img_tensor.to(device)

    print(f"Input image shape: {img_tensor.shape}")

    # C. Extract Features & Compare

    import time

    # 1. Baseline Run
    print("\n--- 1. Baseline Run (No Compile) ---")

    # Warmup
    _ = extractor(img_tensor)

    # Measure
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    for _ in range(5):
        cls_token_baseline, patch_features_baseline = extractor(img_tensor)
    torch.cuda.synchronize() if device == "cuda" else None
    baseline_time = (time.time() - start_time) / 5.0
    print(f"Baseline Inference Time (avg 5 runs): {baseline_time:.4f}s")

    # 2. Compiled Run
    print("\n--- 2. Compiled Run (torch.compile) ---")

    try:
        print("Compiling model (this might take a while)...")
        extractor.model = torch.compile(extractor.model)

        # Warmup (triggers compilation)
        print("Warmup run (compiling)...")
        start_w = time.time()
        _ = extractor(img_tensor)
        print(f"Compilation + Warmup time: {time.time() - start_w:.4f}s")

        # Measure
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        for _ in range(5):
            cls_token_compiled, patch_features_compiled = extractor(img_tensor)
        torch.cuda.synchronize() if device == "cuda" else None
        compiled_time = (time.time() - start_time) / 5.0
        print(f"Compiled Inference Time (avg 5 runs): {compiled_time:.4f}s")

        # Speedup
        print(f"Speedup: {baseline_time / compiled_time:.2f}x")

        # 3. Correctness Check
        print("\n--- 3. Correctness Check ---")
        # Compare CLS tokens
        diff_cls = (cls_token_baseline - cls_token_compiled).abs()
        max_diff_cls = diff_cls.max().item()
        print(f"Max difference in CLS token: {max_diff_cls:.6f}")

        # Compare Patch features
        diff_patch = (patch_features_baseline - patch_features_compiled).abs()
        max_diff_patch = diff_patch.max().item()
        print(f"Max difference in Patch features: {max_diff_patch:.6f}")

        if max_diff_patch < 1e-4:
            print("SUCCESS: Outputs match within tolerance.")
        else:
            print("WARNING: Large difference detected!")

        patch_features = patch_features_compiled

    except Exception as e:
        print(f"Compilation failed or ran into error: {e}")
        patch_features = patch_features_baseline

    print(f"Final extracted feature map shape: {patch_features.shape}")

    # D. Visualize and Save
    if SKLEARN_AVAILABLE:
        visualize_and_save(
            feature_map=patch_features,
            original_img=img_tensor,
            output_path="sklearn_china_dino_viz.png",
        )
    else:
        print("Skipped visualization due to missing sklearn.")

    print("Done.")
