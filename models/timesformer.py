import torch
import torch.nn as nn
from transformers import TimesformerConfig, TimesformerModel, TimesformerForVideoClassification

# ImageNet normalization for pretrained TimeSformer
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = [
    "TimesformerConfig",
    "TimesformerModel",
    "TimesformerForVideoClassification",
    "TimeSformerFeatureExtractor",
    "extract_video_features",
    "video_features_from_tensor",
]


def _normalize_imagenet(x):
    """x: (B, T, C, H, W) in [0, 1]. Normalize with ImageNet stats (C is dim 2)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 1, -1, 1, 1)
    return (x - mean) / std


class TimeSformerFeatureExtractor(nn.Module):
    """
    TimeSformer video feature extractor. Use as a submodule in other models.

    Input: video (B, T, C, H, W), float in [0, 1].
    Output: "sequence" -> (B, T, feat_dim); "patch" -> (B, T, patch_num, feat_dim).
    """

    def __init__(
        self,
        model=None,
        pretrained=None,
        config=None,
        output_format="sequence",
        normalize=True,
    ):
        super().__init__()
        if model is not None:
            self.model = model
        elif pretrained is not None:
            self.model = TimesformerModel.from_pretrained(pretrained)
        elif config is not None:
            self.model = TimesformerModel(config)
        else:
            raise ValueError("Provide one of model, pretrained, or config")
        self.output_format = output_format
        self.normalize = normalize
        self._feat_dim = self.model.config.hidden_size
        self._patch_size = getattr(self.model.config, "patch_size", 16)

    @property
    def feat_dim(self):
        return self._feat_dim

    def forward(self, video):
        """
        Args:
            video: (B, T, C, H, W), float in [0, 1].

        Returns:
            (B, T, feat_dim) if output_format=="sequence",
            (B, T, patch_num, feat_dim) if output_format=="patch".
        """
        B, T, C, H, W = video.shape
        x = video
        if self.normalize:
            x = _normalize_imagenet(x)

        out = self.model(pixel_values=x, return_dict=True)
        last_hidden = out.last_hidden_state

        patch_per_side = H // self._patch_size
        patch_num = patch_per_side * patch_per_side
        feat_dim = last_hidden.size(-1)

        hidden = last_hidden[:, 1:, :]
        hidden = hidden.view(B, T, patch_num, feat_dim)

        if self.output_format == "sequence":
            return hidden.mean(dim=2)
        if self.output_format == "patch":
            return hidden
        raise ValueError('output_format must be "sequence" or "patch"')


def video_features_from_tensor(
    model,
    video,
    output_format="sequence",
    normalize=True,
):
    """
    Extract TimeSformer features from a raw video tensor.

    Args:
        model: TimesformerModel (e.g. from_pretrained("facebook/timesformer-base-finetuned-k400")).
        video: Tensor (B, T, C, H, W), float in [0, 1], following Transformer convention.
        output_format: "sequence" -> [B, seq_len, feat_dim] (one vector per frame, mean over patches);
                       "patch" -> [B, seq_len, patch_num, feat_dim].
        normalize: If True, apply ImageNet normalization before the model (required for pretrained).

    Returns:
        features: [B, seq_len, feat_dim] or [B, seq_len, patch_num, feat_dim].
    """
    B, T, C, H, W = video.shape
    x = video
    if normalize:
        x = _normalize_imagenet(x)

    with torch.no_grad():
        out = model(pixel_values=x, return_dict=True)
    last_hidden = out.last_hidden_state

    cfg = model.config
    patch_size = getattr(cfg, "patch_size", 16)
    patch_per_side = H // patch_size
    patch_num = patch_per_side * patch_per_side

    feat_dim = last_hidden.size(-1)
    hidden = last_hidden[:, 1:, :]
    hidden = hidden.view(B, T, patch_num, feat_dim)

    if output_format == "sequence":
        return hidden.mean(dim=2)
    if output_format == "patch":
        return hidden
    raise ValueError('output_format must be "sequence" or "patch"')


def extract_video_features(model, pixel_values, pool="none"):
    """
    Run TimeSformer and return video features.

    Args:
        model: TimesformerModel instance (use this for feature extraction, not
            TimesformerForVideoClassification).
        pixel_values: Tensor (B, T, C, H, W), following Transformer convention.
        pool: "none" -> return full sequence (B, seq_len, hidden_size);
              "mean" -> mean over sequence (B, hidden_size).

    Returns:
        features: last_hidden_state, or pooled tensor if pool != "none".
    """
    with torch.no_grad():
        out = model(pixel_values=pixel_values, return_dict=True)
    last_hidden = out.last_hidden_state
    if pool == "none":
        return last_hidden
    if pool == "mean":
        return last_hidden.mean(dim=1)
    raise ValueError('pool must be "none" or "mean"')


if __name__ == "__main__":
    import contextlib
    import io
    import logging
    import os
    
    logging.getLogger("transformers").setLevel(logging.ERROR)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        extractor = TimeSformerFeatureExtractor(
            pretrained="facebook/timesformer-base-finetuned-k400",
            output_format="sequence",
            normalize=True,
        )
    video = torch.rand(2, 8, 3, 224, 224)  # (B, T, C, H, W)

    feat = extractor(video)  # (2, 8, 768)
    print("sequence", feat.shape)

    extractor.output_format = "patch"
    feat = extractor(video)  # (2, 8, 196, 768)
    print("patch", feat.shape)