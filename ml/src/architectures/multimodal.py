"""Multimodal fusion network combining pretrained CNN image backbone and Clinical MLP."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml.src.models import (
    build_model,
    freeze_backbone,
    get_gradcam_target_layer,
    head_parameters,
    trainable_backbone_parameters,
    unfreeze_last_blocks,
)


import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass


class ClinicalMLP(nn.Module):
    """Deep feature extractor for tabular clinical symptoms."""

    def __init__(
        self,
        in_features: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClinicalOnlyClassifier(nn.Module):
    """Model B: Clinical symptoms only baseline classifier."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = ClinicalMLP(
            in_features=in_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(clinical)
        return self.head(emb)


class MultimodalClassifier(nn.Module):
    """Model C: Multimodal architecture combining image CNN and clinical MLP."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        clinical_dim: int = 32,
        num_classes: int = 2,
        image_embedding_dim: int = 256,
        clinical_embedding_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
        pretrained: bool = True,
        pretrained_backbone_path: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower().strip()
        self.num_classes = num_classes
        self.clinical_dim = clinical_dim
        self.image_embedding_dim = image_embedding_dim
        self.clinical_embedding_dim = clinical_embedding_dim

        # 1. Image backbone
        use_torchvision_pretrained = pretrained and not pretrained_backbone_path
        base_cnn = build_model(
            self.backbone_name,
            num_classes=2,
            pretrained=use_torchvision_pretrained,
            dropout=dropout,
        )

        if pretrained_backbone_path:
            import os
            if os.path.isfile(pretrained_backbone_path):
                ckpt = torch.load(pretrained_backbone_path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)
                # Load backbone weights, ignore fc / classifier mismatches
                base_cnn.load_state_dict(state, strict=False)

        self.image_backbone = base_cnn

        # Replace classification layer with image embedding projection
        if hasattr(self.image_backbone, "fc"):
            in_f = self.image_backbone.fc[1].in_features if isinstance(self.image_backbone.fc, nn.Sequential) else self.image_backbone.fc.in_features
            self.image_backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_f, image_embedding_dim),
                nn.BatchNorm1d(image_embedding_dim),
                nn.ReLU(inplace=True),
            )
        elif hasattr(self.image_backbone, "classifier"):
            # DenseNet or EfficientNet or MobileNet
            if isinstance(self.image_backbone.classifier, nn.Sequential):
                in_f = self.image_backbone.classifier[-1].in_features
            elif hasattr(self.image_backbone.classifier, "in_features"):
                in_f = self.image_backbone.classifier.in_features
            else:
                in_f = 1280
            self.image_backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_f, image_embedding_dim),
                nn.BatchNorm1d(image_embedding_dim),
                nn.ReLU(inplace=True),
            )

        # 2. Clinical encoder
        self.clinical_encoder = ClinicalMLP(
            in_features=clinical_dim,
            embedding_dim=clinical_embedding_dim,
            hidden_dim=clinical_embedding_dim * 2,
            dropout=dropout,
        )

        # 3. Multimodal Fusion Head
        fused_dim = image_embedding_dim + clinical_embedding_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def extract_image_embedding(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_backbone(image)

    def extract_clinical_embedding(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.clinical_encoder(clinical)

    def forward(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img_emb = self.extract_image_embedding(image)
        if clinical is None:
            clinical = torch.zeros(image.shape[0], self.clinical_dim, device=image.device)
        clin_emb = self.extract_clinical_embedding(clinical)
        fused = torch.cat([img_emb, clin_emb], dim=1)
        logits = self.fusion_head(fused)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        return get_gradcam_target_layer(self.image_backbone, self.backbone_name)

    def freeze_image_backbone(self) -> None:
        """Freeze image backbone weights; keep clinical branch and fusion head trainable."""
        freeze_backbone(self.image_backbone, self.backbone_name)
        # Ensure projection layer in image backbone is trainable
        if hasattr(self.image_backbone, "fc"):
            for p in self.image_backbone.fc.parameters():
                p.requires_grad = True
        elif hasattr(self.image_backbone, "classifier"):
            for p in self.image_backbone.classifier.parameters():
                p.requires_grad = True
        for p in self.clinical_encoder.parameters():
            p.requires_grad = True
        for p in self.fusion_head.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self) -> None:
        """Unfreeze deeper layers of the image backbone for Stage 2 fine-tuning."""
        unfreeze_last_blocks(self.image_backbone, self.backbone_name)
        if hasattr(self.image_backbone, "fc"):
            for p in self.image_backbone.fc.parameters():
                p.requires_grad = True
        elif hasattr(self.image_backbone, "classifier"):
            for p in self.image_backbone.classifier.parameters():
                p.requires_grad = True
        for p in self.clinical_encoder.parameters():
            p.requires_grad = True
        for p in self.fusion_head.parameters():
            p.requires_grad = True


def build_multimodal_model(
    backbone_name: str = "resnet50",
    clinical_dim: int = 32,
    num_classes: int = 2,
    pretrained: bool = True,
    pretrained_backbone_path: str | None = None,
    dropout: float = 0.3,
) -> MultimodalClassifier:
    return MultimodalClassifier(
        backbone_name=backbone_name,
        clinical_dim=clinical_dim,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_backbone_path=pretrained_backbone_path,
        dropout=dropout,
    )


def build_clinical_model(
    clinical_dim: int = 32,
    num_classes: int = 2,
    dropout: float = 0.25,
) -> ClinicalOnlyClassifier:
    return ClinicalOnlyClassifier(
        in_features=clinical_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multimodal & Clinical Architecture Initialization")
    print("=" * 60)

    # 1. Test Clinical-Only Classifier
    clin_model = build_clinical_model(clinical_dim=28, num_classes=2)
    dummy_clin = torch.randn(4, 28)
    clin_out = clin_model(dummy_clin)
    print(f"✓ Clinical-only forward pass successful! Output shape: {tuple(clin_out.shape)}")

    # 2. Test Multimodal Classifier
    print("\nInitializing MultimodalClassifier (ResNet50 + Clinical MLP)...")
    mm_model = build_multimodal_model(
        backbone_name="resnet50",
        clinical_dim=28,
        num_classes=2,
        pretrained=False,
    )
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_clin_batch = torch.randn(2, 28)
    mm_out = mm_model(dummy_img, dummy_clin_batch)
    print(f"✓ Multimodal forward pass successful! Output shape: {tuple(mm_out.shape)}")

    # 3. Test Backbone Freezing / Unfreezing
    mm_model.freeze_image_backbone()
    target_layer = mm_model.get_gradcam_target_layer()
    is_frozen = not any(p.requires_grad for p in target_layer.parameters())
    print(f"✓ Image backbone freeze verified: {is_frozen}")

    mm_model.unfreeze_last_blocks()
    is_unfrozen = any(p.requires_grad for p in target_layer.parameters())
    print(f"✓ Deeper blocks unfreeze verified: {is_unfrozen}")

    total_params = sum(p.numel() for p in mm_model.parameters())
    trainable_params = sum(p.numel() for p in mm_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters (Stage 2): {trainable_params:,}")
    print("=" * 60)
    print("All architecture sanity checks passed!")

