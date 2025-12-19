import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.models.video as video_models
from PIL import Image
import numpy as np


# -------------------------
# 2D ResNet features
# -------------------------

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True):
        """
        2D ResNet feature extractor.
        Supports resnet50 / resnet101 with ImageNet weights.
        """
        super().__init__()
        if model_name == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = resnet.fc.in_features
        elif model_name == "resnet101":
            resnet = models.resnet101(
                weights=models.ResNet101_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = resnet.fc.in_features
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # Remove final classification layer
        resnet.fc = nn.Identity()
        self.backbone = resnet.eval()

        # Image preprocessing
        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def forward(self, frames):
        """
        Args:
            frames: list of numpy arrays or PIL Images (RGB).
        Returns:
            [T, D] tensor of frame features.
        """
        if not isinstance(frames, (list, tuple)):
            raise TypeError("frames must be a list/tuple of numpy arrays or PIL Images")

        imgs = torch.stack([self.transform(_to_pil(f)) for f in frames])

        device = next(self.backbone.parameters()).device
        imgs = imgs.to(device)
        features = self.backbone(imgs)  # [T, D]
        return features


# -------------------------
# 2D CLIP ViT features
# -------------------------

class CLIPViTFeatureExtractor(nn.Module):
    """
    CLIP ViT-based feature extractor.
    Uses CLIP's vision transformer for image features.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        use_fast: bool = True,
    ):
        super().__init__()
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError(
                "transformers is required for CLIP. Install with: pip install transformers"
            )

        # CLIP vision encoder
        self.model = CLIPModel.from_pretrained(model_name)

        # Fast image processor (avoids slow-processor warning)
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            use_fast=use_fast,
        )
        self.model.eval()

        # For ViT-B/32 this is 512
        self.feature_dim = self.model.config.projection_dim

    @torch.no_grad()
    def forward(self, frames):
        """
        Args:
            frames: list of numpy arrays or PIL Images (RGB).
        Returns:
            [T, D] tensor of frame features.
        """
        if not isinstance(frames, (list, tuple)):
            raise TypeError("frames must be a list/tuple of numpy arrays or PIL Images")

        pil_frames = []
        for f in frames:
            if isinstance(f, np.ndarray):
                arr = f
                # Allow CHW or HWC input
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
                    raise ValueError(f"Unsupported image shape for CLIP: {arr.shape}")
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                if arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                pil_frames.append(Image.fromarray(arr))
            else:
                pil_frames.append(_to_pil(f))

        inputs = self.processor(images=pil_frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)

        features = self.model.get_image_features(pixel_values=pixel_values)  # [T, D]
        return features


# -------------------------
# Utility: factory
# -------------------------

def create_feature_extractor(backbone: str = "resnet50", **kwargs):
    """
    Factory for feature extractors.
    """
    dispatch = {
        "resnet50": lambda: ResNetFeatureExtractor(model_name="resnet50", **kwargs),
        "resnet101": lambda: ResNetFeatureExtractor(model_name="resnet101", **kwargs),
        "clip-vit": lambda: CLIPViTFeatureExtractor(
            model_name=kwargs.get("model_name", "openai/clip-vit-base-patch32"),
            use_fast=kwargs.get("use_fast", True),
        ),
        "clip": lambda: CLIPViTFeatureExtractor(
            model_name=kwargs.get("model_name", "openai/clip-vit-base-patch32"),
            use_fast=kwargs.get("use_fast", True),
        ),
        "swin3d": lambda: VideoSwinFeatureExtractor(**kwargs),
        "video-swin": lambda: VideoSwinFeatureExtractor(**kwargs),
    }
    if backbone not in dispatch:
        raise ValueError(
            "Unsupported backbone. Choose from: "
            "resnet50, resnet101, clip-vit, clip, swin3d, video-swin"
        )
    return dispatch[backbone]()


def _to_pil(frame):
    """Convert frame (numpy/PIL) to PIL.Image."""
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    raise TypeError("Frame must be a numpy array or PIL Image")


# -------------------------
# 3D Swin video features
# -------------------------

class VideoSwinFeatureExtractor(nn.Module):
    """
    Video Swin Transformer (3D) feature extractor.

    We explicitly take features BEFORE the final global pooling:
        [B, T', H', W', C] -> avg over H', W' -> [B, T', C]
    so that we keep a temporal feature sequence instead of a single clip vector.
    """

    def __init__(self, model_name: str = "swin3d_t", pretrained: bool = True):
        super().__init__()
        if model_name != "swin3d_t":
            raise ValueError("Supported 3D backbone: swin3d_t")

        weights = video_models.Swin3D_T_Weights.DEFAULT if pretrained else None
        model = video_models.swin3d_t(weights=weights)

        # Remove classification head (we don't need logits)
        model.head = nn.Identity()
        self.backbone = model.eval()

        # Dim of last stage features
        self.feature_dim = self.backbone.num_features

        # Basic video preprocessing (roughly aligned with Kinetics-style)
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def forward(self, frames):
        """
        Args:
            frames: list of numpy arrays or PIL Images (RGB).
        Returns:
            [T', D] tensor of 3D temporal features (T' is downsampled in time).
        """
        if not isinstance(frames, (list, tuple)):
            raise TypeError("frames must be a list/tuple of numpy arrays or PIL Images")

        # [T, C, H, W]
        imgs = torch.stack([self.transform(_to_pil(f)) for f in frames])

        # Swin3D expects [B, C, T, H, W]
        x = imgs.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        device = next(self.backbone.parameters()).device
        x = x.to(device)

        # ---- Explicitly run backbone up to normalized feature map ----
        # 1) patch embedding
        x = self.backbone.patch_embed(x)   # [B, T', H', W', C]
        # 2) dropout
        x = self.backbone.pos_drop(x)
        # 3) transformer stages
        x = self.backbone.features(x)      # [B, T', H', W', C]
        # 4) final norm
        x = self.backbone.norm(x)          # [B, T', H', W', C]

        # ---- Average over spatial dims, keep temporal tokens ----
        B, T_p, H_p, W_p, C = x.shape
        x = x.view(B, T_p, H_p * W_p, C).mean(dim=2)  # [B, T', C]

        # Remove batch dim and move to CPU: [T', D]
        x = x.squeeze(0).cpu()

        return x
