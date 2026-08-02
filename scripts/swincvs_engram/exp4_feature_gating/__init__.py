"""
Experiment 4: Feature Gating for CVS Classification

Instead of concatenating masks as input channels (Exp 2-3), we:
1. Keep SwinV2 as standard 3-channel RGB input
2. Process masks through a separate MaskEncoder CNN
3. Use mask encoding to gate SwinV2 features before LSTM

Architecture:
    frames (B, 5, 3, 384, 384) -> SwinV2 -> features (B, 5, 1024)
                                                  |
    masks (B, 5, 2, 384, 384) -> MaskEncoder -> gate (B, 5, 1024)
                                                  |
                                    features * sigmoid(gate)
                                                  |
                                              LSTM -> (B, 3)

Hypothesis: Feature-level gating allows the model to learn which SwinV2
features are relevant given the spatial mask information, without modifying
the pretrained backbone's input processing.
"""

__version__ = "0.1.0"
__experiment__ = "E4"
