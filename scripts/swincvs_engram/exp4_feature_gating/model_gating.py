"""
Feature Gating Model for CVS Classification

MaskEncoder: CNN that encodes spatial masks into feature-space gate weights
SwinCVSModelWithGating: Wrapper that applies mask-based gating to SwinV2 features
"""

import torch
import torch.nn as nn


class MaskEncoder(nn.Module):
    """
    Encode spatial masks into feature-space gate weights.

    Input: (B, 2, H, W) - GB and Tool masks (soft confidence values)
    Output: (B, feature_dim) - gate weights for SwinV2 features

    Architecture matches SwinV2's spatial reduction:
    - 384x384 -> 96x96 -> 24x24 -> 12x12 -> 1x1 (global pool)
    """

    def __init__(self, feature_dim=1024, input_channels=2):
        super().__init__()
        self.feature_dim = feature_dim

        # CNN to process masks - mirrors SwinV2's spatial reduction
        self.conv = nn.Sequential(
            # 384 -> 96 (stride 4, like patch_embed)
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 96 -> 24 (stride 4)
            nn.Conv2d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 24 -> 12 (stride 2)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 12 -> 1 (global pool)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Project to feature dimension
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, masks):
        """
        Args:
            masks: (B, 2, H, W) - GB mask (channel 0) and Tool mask (channel 1)

        Returns:
            gate: (B, feature_dim) - gate weights
        """
        x = self.conv(masks)  # (B, 128)
        gate = self.fc(x)      # (B, feature_dim)
        return gate


class SwinCVSModelWithGating(nn.Module):
    """
    SwinCVS model with feature gating based on mask information.

    The SwinV2 backbone processes RGB frames normally.
    A separate MaskEncoder processes the segmentation masks.
    The mask encoding gates the SwinV2 features before LSTM.
    """

    def __init__(self, swinv2_model, lstm, fc_lstm, feature_dim=1024,
                 lstm_hidden_size=256, num_classes=3,
                 freeze_backbone=True, freeze_lstm=False):
        super().__init__()

        # Original model components
        self.swinv2_model = swinv2_model
        self.lstm = lstm
        self.fc_lstm = fc_lstm

        # New mask encoder (trainable)
        self.mask_encoder = MaskEncoder(feature_dim=feature_dim)

        # Store dimensions
        self.feature_dim = feature_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes

        # Freeze settings
        self.freeze_backbone = freeze_backbone
        self.freeze_lstm = freeze_lstm

        if freeze_backbone:
            for param in self.swinv2_model.parameters():
                param.requires_grad = False

        if freeze_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.fc_lstm.parameters():
                param.requires_grad = False

    def forward(self, frames, masks):
        """
        Args:
            frames: (B, T, 3, H, W) - RGB frames, T=5 for LSTM
            masks: (B, T, 2, H, W) - GB and Tool masks for each frame

        Returns:
            output: (B, num_classes) - classification logits
        """
        B, T, C, H, W = frames.shape

        # Flatten batch and time for SwinV2
        frames_flat = frames.view(B * T, C, H, W)

        # Process frames through SwinV2
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.swinv2_model.forward_features(frames_flat)  # (B*T, feature_dim)
        else:
            features = self.swinv2_model.forward_features(frames_flat)

        # Process masks through encoder
        masks_flat = masks.view(B * T, 2, H, W)
        gate = self.mask_encoder(masks_flat)  # (B*T, feature_dim)

        # Apply gating with sigmoid
        gated_features = features * torch.sigmoid(gate)  # (B*T, feature_dim)

        # Reshape for LSTM: (B, T, feature_dim)
        gated_features = gated_features.view(B, T, -1)

        # LSTM processing
        if self.freeze_lstm:
            with torch.no_grad():
                lstm_out, _ = self.lstm(gated_features)
        else:
            lstm_out, _ = self.lstm(gated_features)

        # Classification from last timestep
        output = self.fc_lstm(lstm_out[:, -1, :])  # (B, num_classes)

        return output

    def get_trainable_params(self):
        """Return list of trainable parameters for optimizer."""
        params = list(self.mask_encoder.parameters())

        if not self.freeze_backbone:
            params += list(self.swinv2_model.parameters())

        if not self.freeze_lstm:
            params += list(self.lstm.parameters())
            params += list(self.fc_lstm.parameters())

        return params

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Component breakdown
        backbone = sum(p.numel() for p in self.swinv2_model.parameters())
        lstm = sum(p.numel() for p in self.lstm.parameters())
        fc = sum(p.numel() for p in self.fc_lstm.parameters())
        mask_enc = sum(p.numel() for p in self.mask_encoder.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'backbone': backbone,
            'lstm': lstm,
            'fc': fc,
            'mask_encoder': mask_enc,
        }


def create_gated_model_from_checkpoint(checkpoint_path, device='cuda',
                                        freeze_backbone=True, freeze_lstm=False):
    """
    Load a pretrained SwinCVS checkpoint and wrap it with feature gating.

    Args:
        checkpoint_path: Path to SwinCVS checkpoint (.pt file)
        device: Device to load model on
        freeze_backbone: Whether to freeze SwinV2 backbone
        freeze_lstm: Whether to freeze LSTM and FC layers

    Returns:
        SwinCVSModelWithGating model
    """
    import sys
    from pathlib import Path

    # Add SwinCVS to path
    swincvs_root = Path(__file__).parent.parent.parent.parent.parent / 'SwinCVS'
    if str(swincvs_root) not in sys.path:
        sys.path.insert(0, str(swincvs_root))
    if str(swincvs_root / 'scripts') not in sys.path:
        sys.path.insert(0, str(swincvs_root / 'scripts'))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine if it's a full model or just state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Build model architecture
    from m_swinv2 import SwinTransformerV2
    from m_swincvs import SwinCVSModel

    # Default SwinCVS config values
    swinv2 = SwinTransformerV2(
        img_size=384,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=24,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6],
    )

    # Extract SwinV2 weights from state dict
    swinv2_state = {}
    lstm_state = {}
    fc_lstm_state = {}

    for key, value in state_dict.items():
        if key.startswith('swinv2_model.'):
            new_key = key.replace('swinv2_model.', '')
            swinv2_state[new_key] = value
        elif key.startswith('lstm.'):
            lstm_state[key] = value
        elif key.startswith('fc_lstm.'):
            fc_lstm_state[key] = value

    # Load SwinV2 weights
    swinv2.load_state_dict(swinv2_state, strict=False)

    # Create LSTM and FC
    feature_dim = swinv2.num_features  # 1024 for base model
    lstm_hidden_size = 256
    num_classes = 3

    lstm = nn.LSTM(
        input_size=feature_dim,
        hidden_size=lstm_hidden_size,
        num_layers=2,
        batch_first=True
    )
    fc_lstm = nn.Linear(lstm_hidden_size, num_classes)

    # Load LSTM and FC weights
    lstm_state_clean = {k.replace('lstm.', ''): v for k, v in lstm_state.items()}
    fc_state_clean = {k.replace('fc_lstm.', ''): v for k, v in fc_lstm_state.items()}

    lstm.load_state_dict(lstm_state_clean)
    fc_lstm.load_state_dict(fc_state_clean)

    # Create gated model
    model = SwinCVSModelWithGating(
        swinv2_model=swinv2,
        lstm=lstm,
        fc_lstm=fc_lstm,
        feature_dim=feature_dim,
        lstm_hidden_size=lstm_hidden_size,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        freeze_lstm=freeze_lstm,
    )

    model = model.to(device)

    return model


if __name__ == '__main__':
    # Test the model
    print("Testing MaskEncoder...")
    encoder = MaskEncoder(feature_dim=1024)
    dummy_masks = torch.randn(2, 2, 384, 384)
    gate = encoder(dummy_masks)
    print(f"  Input: {dummy_masks.shape}")
    print(f"  Output: {gate.shape}")

    print("\nMaskEncoder parameters:")
    total = sum(p.numel() for p in encoder.parameters())
    print(f"  Total: {total:,} ({total/1e6:.2f}M)")
