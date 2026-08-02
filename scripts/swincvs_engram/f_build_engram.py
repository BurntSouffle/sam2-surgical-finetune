"""
E1 Experiment: Modified Model Builder for Visual Engrams

This module extends the original SwinCVS model builder to handle 5-channel input:
- Channels 0-2: RGB (ImageNet pretrained weights preserved)
- Channels 3-4: GB mask + Tool mask (initialized with small random values)

Key modification: The patch embedding layer (patch_embed.proj) is extended from
[embed_dim, 3, patch_size, patch_size] to [embed_dim, 5, patch_size, patch_size]
while preserving the pretrained RGB weights.

Usage:
    from f_build_engram import build_engram_model
    model = build_engram_model(config)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add SwinCVS scripts to path for imports
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent.parent  # Go up to DISSERTATION level
SWINCVS_ROOT = PROJECT_ROOT / 'SwinCVS'

sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT / 'scripts'))

from scripts.m_swinv2 import SwinTransformerV2
from scripts.m_swincvs import SwinCVSModel


# =============================================================================
# MODIFIED SWINCVS MODEL FOR VARIABLE INPUT CHANNELS
# =============================================================================

class SwinCVSModelEngram(nn.Module):
    """
    Modified SwinCVSModel that handles variable input channels (e.g., 5 for engram).

    The original SwinCVSModel hardcodes `x = x.view(-1, 3, 384, 384)` in forward().
    This version dynamically reads the number of input channels from the backbone.

    This is the only change from the original SwinCVSModel - everything else is identical.
    """

    def __init__(self, swinv2_model, config, num_classes=3):
        super(SwinCVSModelEngram, self).__init__()
        self.swinv2_model = swinv2_model
        self.backbone = swinv2_model  # Alias for compatibility with test scripts
        self.lstm_hidden_size = config.MODEL.LSTM_PARAMS.HIDDEN_SIZE
        self.num_classes = num_classes

        # Get input channels from the backbone's patch embedding
        self.in_chans = swinv2_model.patch_embed.proj.in_channels
        self.img_size = 384  # SwinCVS uses 384x384 input

        # Toggle for additional classifier after the backbone
        if config.MODEL.E2E != True:
            self.multiclassifier = False
        else:
            self.multiclassifier = config.MODEL.MULTICLASSIFIER
        self.inference = config.MODEL.INFERENCE  # Toggle for inference mode

        # LSTM for temporal sequence processing
        self.lstm = nn.LSTM(
            input_size=self.swinv2_model.num_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=config.MODEL.LSTM_PARAMS.NUM_LAYERS,
            batch_first=True
        )

        # Fully connected layer for classification after LSTM
        self.fc_lstm = nn.Linear(self.lstm_hidden_size, num_classes)
        if self.multiclassifier:
            # New fully connected layer for mid-stream classification (SwinV2 feature classification)
            self.fc_swin = nn.Linear(self.swinv2_model.num_features, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len=5, in_chans, 384, 384)
        # For engram: in_chans=5 (RGB + GB + Tool masks)
        batch_size, seq_len, _, _, _ = x.size()

        # Reshape input for SwinV2 - DYNAMIC in_chans instead of hardcoded 3
        x = x.view(-1, self.in_chans, self.img_size, self.img_size)

        # Extract features from SwinV2
        features = self.swinv2_model.forward_features(x)  # Shape: (batch_size * seq_len, num_features)

        # Optional mid-stream classification
        if self.multiclassifier and not self.inference:
            swin_classification = self.fc_swin(features)  # Shape: (batch_size * seq_len, num_classes)
            swin_classification = swin_classification.view(batch_size, seq_len, -1)  # Reshape for sequence output

        # Reshape back to (batch_size, seq_len, num_features)
        features = features.view(batch_size, seq_len, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # Shape: (batch_size, seq_len, hidden_size)

        # Use the last time step's output from LSTM for classification
        lstm_classification = self.fc_lstm(lstm_out[:, -1, :])  # Shape: (batch_size, num_classes)

        if self.multiclassifier and not self.inference:
            return swin_classification[:, -1, :], lstm_classification
        else:
            return lstm_classification


# =============================================================================
# WEIGHT EXTENSION FOR 5-CHANNEL INPUT
# =============================================================================

def extend_patch_embed_weights(
    pretrained_weight: torch.Tensor,
    target_in_chans: int = 5,
    init_std: float = 0.02,
    init_strategy: str = 'random',
) -> torch.Tensor:
    """
    Extend patch embedding weights from 3 channels to target_in_chans.

    Args:
        pretrained_weight: Original weights [embed_dim, 3, patch_h, patch_w]
        target_in_chans: Target number of input channels (default: 5)
        init_std: Standard deviation for random initialization
        init_strategy: Initialization strategy for new channels:
            - 'random': Small random values (N(0, init_std))
            - 'zero': Zero initialization
            - 'copy_mean': Copy mean of RGB channels

    Returns:
        Extended weights [embed_dim, target_in_chans, patch_h, patch_w]
    """
    embed_dim, orig_chans, patch_h, patch_w = pretrained_weight.shape

    if orig_chans >= target_in_chans:
        print(f"  [WARN] Pretrained has {orig_chans} channels, target is {target_in_chans}. No extension needed.")
        return pretrained_weight[:, :target_in_chans, :, :]

    new_chans = target_in_chans - orig_chans

    print(f"  Extending patch_embed.proj weights:")
    print(f"    Original shape: [{embed_dim}, {orig_chans}, {patch_h}, {patch_w}]")
    print(f"    Target shape:   [{embed_dim}, {target_in_chans}, {patch_h}, {patch_w}]")
    print(f"    New channels:   {new_chans} (for mask inputs)")
    print(f"    Init strategy:  {init_strategy}")

    # Initialize new channel weights based on strategy
    if init_strategy == 'random':
        # Small random values - allows model to learn from masks
        new_weights = torch.randn(embed_dim, new_chans, patch_h, patch_w) * init_std
        print(f"    Init values:    N(0, {init_std}) random")

    elif init_strategy == 'zero':
        # Zero initialization - masks have no initial effect
        new_weights = torch.zeros(embed_dim, new_chans, patch_h, patch_w)
        print(f"    Init values:    zeros")

    elif init_strategy == 'copy_mean':
        # Copy mean of RGB channels - provides reasonable starting point
        rgb_mean = pretrained_weight.mean(dim=1, keepdim=True)  # [embed_dim, 1, h, w]
        new_weights = rgb_mean.repeat(1, new_chans, 1, 1)
        print(f"    Init values:    mean of RGB channels")

    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")

    # Concatenate along channel dimension
    extended_weight = torch.cat([pretrained_weight, new_weights], dim=1)

    print(f"    Final shape:    {list(extended_weight.shape)}")

    return extended_weight


def load_pretrained_engram(config, model, init_std: float = 0.02, init_strategy: str = 'random'):
    """
    Load pretrained ImageNet weights with extension for 5-channel input.

    This is a modified version of load_pretrained() from m_swinv2.py that:
    1. Loads the 3-channel pretrained weights
    2. Extends patch_embed.proj weights to 5 channels
    3. Preserves RGB weights, initializes mask channels

    Args:
        config: Configuration object with BACKBONE settings
        model: SwinTransformerV2 model with 5-channel input
        init_std: Std for random initialization of new channels
        init_strategy: How to initialize new channels ('random', 'zero', 'copy_mean')
    """
    print(f"\n{'='*70}")
    print("LOADING PRETRAINED WEIGHTS WITH ENGRAM EXTENSION")
    print(f"{'='*70}")

    weights_path = SWINCVS_ROOT / 'weights' / config.BACKBONE.PRETRAINED
    print(f"Loading backbone weights from: {weights_path}")

    if not weights_path.exists():
        # Try relative path
        weights_path = Path('weights') / config.BACKBONE.PRETRAINED
        if not weights_path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {config.BACKBONE.PRETRAINED}")

    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model']

    # Get model's expected input channels
    model_in_chans = model.patch_embed.proj.in_channels
    pretrained_in_chans = state_dict['patch_embed.proj.weight'].shape[1]

    print(f"\nChannel configuration:")
    print(f"  Pretrained model: {pretrained_in_chans} channels (RGB)")
    print(f"  Target model:     {model_in_chans} channels (RGB + masks)")

    # ==========================================================================
    # EXTEND PATCH EMBEDDING WEIGHTS (key modification for E1)
    # ==========================================================================
    if model_in_chans > pretrained_in_chans:
        print(f"\n[ENGRAM] Extending patch embedding from {pretrained_in_chans} to {model_in_chans} channels...")

        # Extend the patch embedding projection weights
        pretrained_proj_weight = state_dict['patch_embed.proj.weight']
        extended_proj_weight = extend_patch_embed_weights(
            pretrained_proj_weight,
            target_in_chans=model_in_chans,
            init_std=init_std,
            init_strategy=init_strategy,
        )
        state_dict['patch_embed.proj.weight'] = extended_proj_weight

        # Bias doesn't need modification (it's per output channel, not input)
        print(f"  patch_embed.proj.bias: kept as-is (shape: {state_dict['patch_embed.proj.bias'].shape})")

    elif model_in_chans < pretrained_in_chans:
        print(f"\n[WARN] Model has fewer channels than pretrained. Truncating...")
        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'][:, :model_in_chans, :, :]

    # ==========================================================================
    # STANDARD SWINV2 WEIGHT PROCESSING (from original load_pretrained)
    # ==========================================================================

    # Delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # Delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del state_dict[k]

    # Delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # Bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"  [WARN] Error in loading {k}, passing...")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2),
                    mode='bicubic'
                )
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # Bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C2:
            print(f"  [WARN] Error in loading {k}, passing...")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic'
                )
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # Check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if Nc1 != Nc2:
        if Nc1 == 21841 and Nc2 == 1000:
            print("  Loading ImageNet-22K weight to ImageNet-1K...")
            map22kto1k_path = SWINCVS_ROOT / 'data' / 'map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"  [WARN] Classifier head mismatch, re-init to 0")

    # Load state dict
    msg = model.load_state_dict(state_dict, strict=False)

    if msg.missing_keys:
        print(f"\n  Missing keys (expected): {msg.missing_keys[:5]}...")
    if msg.unexpected_keys:
        print(f"  Unexpected keys: {msg.unexpected_keys[:5]}...")

    print(f"\n{'='*70}")
    print("PRETRAINED WEIGHTS LOADED SUCCESSFULLY WITH ENGRAM EXTENSION")
    print(f"{'='*70}\n")

    del checkpoint
    torch.cuda.empty_cache()


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_engram_model(config):
    """
    Build SwinCVS model with 5-channel input for E1 (Visual Engrams) experiment.

    This is a modified version of build_model() from f_build.py that:
    1. Creates SwinTransformerV2 with 5 input channels
    2. Loads pretrained weights with channel extension
    3. Wraps in SwinCVSModel for LSTM temporal modeling

    Args:
        config: Configuration object with model settings

    Returns:
        SwinCVSModel with 5-channel input backbone
    """
    print(f"\n{'#'*70}")
    print("# E1 EXPERIMENT: BUILDING SWINCVS WITH VISUAL ENGRAMS")
    print(f"{'#'*70}")

    # Validate engram configuration
    engram_enabled = hasattr(config, 'ENGRAM') and config.ENGRAM.get('ENABLED', False)
    in_chans = config.BACKBONE.SWINV2.IN_CHANS

    print(f"\nConfiguration:")
    print(f"  ENGRAM.ENABLED:  {engram_enabled}")
    print(f"  IN_CHANS:        {in_chans}")
    print(f"  LSTM:            {config.MODEL.LSTM}")
    print(f"  E2E:             {config.MODEL.E2E}")
    print(f"  MULTICLASSIFIER: {config.MODEL.MULTICLASSIFIER}")

    if engram_enabled and in_chans != 5:
        print(f"\n  [WARN] ENGRAM enabled but IN_CHANS={in_chans}. Expected 5.")

    if not engram_enabled and in_chans == 5:
        print(f"\n  [WARN] IN_CHANS=5 but ENGRAM not enabled. Enabling engram weight extension.")
        engram_enabled = True

    # ==========================================================================
    # INITIALIZE BACKBONE (5-channel SwinTransformerV2)
    # ==========================================================================
    print(f"\nInitializing SwinTransformerV2 backbone...")

    model = SwinTransformerV2(
        img_size=384,  # Important for correct ImageNet pretrained weights
        patch_size=config.BACKBONE.SWINV2.PATCH_SIZE,
        in_chans=config.BACKBONE.SWINV2.IN_CHANS,  # 5 for engram
        num_classes=config.BACKBONE.NUM_CLASSES,
        embed_dim=config.BACKBONE.SWINV2.EMBED_DIM,
        depths=config.BACKBONE.SWINV2.DEPTHS,
        num_heads=config.BACKBONE.SWINV2.NUM_HEADS,
        window_size=config.BACKBONE.SWINV2.WINDOW_SIZE,
        mlp_ratio=config.BACKBONE.SWINV2.MLP_RATIO,
        qkv_bias=config.BACKBONE.SWINV2.QKV_BIAS,
        drop_rate=config.BACKBONE.DROP_RATE,
        drop_path_rate=config.BACKBONE.DROP_PATH_RATE,
        ape=config.BACKBONE.SWINV2.APE,
        patch_norm=config.BACKBONE.SWINV2.PATCH_NORM,
        use_checkpoint=config.BACKBONE.USE_CHECKPOINT,
        pretrained_window_sizes=config.BACKBONE.SWINV2.PRETRAINED_WINDOW_SIZES,
    )

    print(f"  Backbone created with {in_chans} input channels")
    print(f"  patch_embed.proj: Conv2d({in_chans}, {config.BACKBONE.SWINV2.EMBED_DIM}, kernel_size=4, stride=4)")

    # ==========================================================================
    # LOAD PRETRAINED WEIGHTS (with channel extension)
    # ==========================================================================
    if not config.MODEL.INFERENCE:
        try:
            if 'swinv2_base_patch4' in config.BACKBONE.PRETRAINED:
                # Use ImageNet pretrained weights with engram extension
                if engram_enabled and in_chans > 3:
                    # Get initialization settings from config if available
                    init_std = 0.02
                    init_strategy = 'random'

                    if hasattr(config, 'ENGRAM'):
                        init_std = config.ENGRAM.get('INIT_STD', 0.02)
                        init_strategy = config.ENGRAM.get('INIT_STRATEGY', 'random')

                    load_pretrained_engram(
                        config, model,
                        init_std=init_std,
                        init_strategy=init_strategy,
                    )
                else:
                    # Standard 3-channel loading
                    from scripts.m_swinv2 import load_pretrained
                    load_pretrained(config, model)

            elif config.BACKBONE.PRETRAINED is not None:
                # Use endoscapes pretrained weights (for frozen model version)
                print(f"\nLoading backbone weight: {config.BACKBONE.PRETRAINED}")
                model.head = nn.Linear(in_features=1024, out_features=3, bias=True)
                weights = SWINCVS_ROOT / 'weights' / config.BACKBONE.PRETRAINED
                model.load_state_dict(torch.load(weights))

            print("Backbone weights loaded successfully!")

        except Exception as e:
            print(f"[ERROR] Failed to load pretrained weights: {e}")
            print("Backbone NOT pretrained!")

    # ==========================================================================
    # CONFIGURE FOR LSTM / CLASSIFICATION
    # ==========================================================================
    if config.MODEL.LSTM:
        # Remove MLP classifier (LSTM will handle classification)
        model.head = nn.Identity()

        if not config.MODEL.E2E:
            # Freeze backbone weights for non-E2E training
            print("\nFreezing backbone weights (E2E=False)...")
            for param in model.parameters():
                param.requires_grad = False

        # Wrap in SwinCVSModelEngram for LSTM temporal modeling
        # Use Engram version which handles variable input channels (5 instead of 3)
        print("\nWrapping backbone in SwinCVSModelEngram (LSTM enabled, variable in_chans)...")
        swincvs = SwinCVSModelEngram(model, config)

        # Print parameter counts
        total_params = sum(p.numel() for p in swincvs.parameters())
        trainable_params = sum(p.numel() for p in swincvs.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total:     {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

        return swincvs
    else:
        # Pure SwinV2 model (no LSTM)
        model.head = nn.Linear(in_features=1024, out_features=3, bias=True)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total:     {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

        return model


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def verify_engram_model(model, config):
    """
    Verify that the model is correctly configured for engram input.

    Args:
        model: Built model (SwinCVSModel or SwinTransformerV2)
        config: Configuration object

    Returns:
        bool: True if model is correctly configured
    """
    print("\n" + "="*50)
    print("VERIFYING ENGRAM MODEL CONFIGURATION")
    print("="*50)

    # Get backbone from SwinCVSModel if wrapped
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        backbone = model

    # Check input channels
    patch_embed_in_chans = backbone.patch_embed.proj.in_channels
    expected_in_chans = config.BACKBONE.SWINV2.IN_CHANS

    print(f"\nPatch embedding input channels:")
    print(f"  Expected: {expected_in_chans}")
    print(f"  Actual:   {patch_embed_in_chans}")

    if patch_embed_in_chans != expected_in_chans:
        print("  [FAIL] Channel mismatch!")
        return False

    print("  [PASS] Channels match")

    # Check weight shapes
    weight_shape = backbone.patch_embed.proj.weight.shape
    expected_shape = (config.BACKBONE.SWINV2.EMBED_DIM, expected_in_chans, 4, 4)

    print(f"\nPatch embedding weight shape:")
    print(f"  Expected: {expected_shape}")
    print(f"  Actual:   {tuple(weight_shape)}")

    if tuple(weight_shape) != expected_shape:
        print("  [FAIL] Shape mismatch!")
        return False

    print("  [PASS] Shape matches")

    # Test forward pass with dummy input
    print("\nTesting forward pass with 5-channel input...")
    try:
        dummy_input = torch.randn(1, expected_in_chans, 384, 384)

        if hasattr(model, 'backbone'):
            # SwinCVSModel expects sequence input
            dummy_input = dummy_input.unsqueeze(0).repeat(1, 5, 1, 1, 1)  # [B, T, C, H, W]

        with torch.no_grad():
            model.eval()
            output = model(dummy_input)

        print(f"  Input shape:  {list(dummy_input.shape)}")
        print(f"  Output shape: {list(output.shape)}")
        print("  [PASS] Forward pass successful")

    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        return False

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE - ALL CHECKS PASSED")
    print("="*50 + "\n")

    return True


# =============================================================================
# MAIN - TEST
# =============================================================================

if __name__ == '__main__':
    """Test the engram model builder."""
    print("\n" + "#"*70)
    print("# TESTING E1 ENGRAM MODEL BUILDER")
    print("#"*70 + "\n")

    # Create mock config for testing
    class MockConfig:
        class BACKBONE:
            PRETRAINED = 'swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
            DROP_PATH_RATE = 0.2
            DROP_RATE = 0.0
            NUM_CLASSES = 1000
            USE_CHECKPOINT = False

            class SWINV2:
                PATCH_SIZE = 4
                IN_CHANS = 5  # 5 channels for engram
                EMBED_DIM = 128
                DEPTHS = [2, 2, 18, 2]
                NUM_HEADS = [4, 8, 16, 32]
                WINDOW_SIZE = 24
                PRETRAINED_WINDOW_SIZES = [12, 12, 12, 6]
                MLP_RATIO = 4
                QKV_BIAS = True
                APE = False
                PATCH_NORM = True

        class MODEL:
            LSTM = True
            E2E = True
            MULTICLASSIFIER = True
            INFERENCE = False

            class LSTM_PARAMS:
                NUM_LAYERS = 2
                HIDDEN_SIZE = 256

        class ENGRAM:
            ENABLED = True
            INIT_STD = 0.02
            INIT_STRATEGY = 'random'

    config = MockConfig()

    print("Test configuration:")
    print(f"  IN_CHANS:        {config.BACKBONE.SWINV2.IN_CHANS}")
    print(f"  ENGRAM.ENABLED:  {config.ENGRAM.ENABLED}")
    print(f"  LSTM:            {config.MODEL.LSTM}")

    print("\nNote: Full test requires pretrained weights file.")
    print("Run from project root with weights available.\n")
