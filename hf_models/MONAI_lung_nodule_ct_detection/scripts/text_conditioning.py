from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import torch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import RetinaNet
from torch import Tensor, nn


LOC_VOCAB = (
    "left upper lobe",
    "left lower lobe",
    "right upper lobe",
    "right middle lobe",
    "right lower lobe",
    "left lung",
    "right lung",
    "upper lobe",
    "lower lobe",
    "lingula",
    "lingula or left lower lobe",
)

POSITION_VOCAB = (
    "apical",
    "superior",
    "basal",
    "inferior",
    "anterior",
    "posterior",
    "medial",
    "lateral",
    "centrilobular",
    "justapleural",
    "peripheral",
)

NODULE_TYPE_VOCAB = (
    "nodule",
    "micronodule",
    "granuloma",
    "mass",
)

CHARACTERISTIC_VOCAB = (
    "texture",
    "calcification",
    "internal structure",
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
)

TEXT_FEATURE_DIM = (
    1
    + 2
    + len(LOC_VOCAB)
    + len(POSITION_VOCAB)
    + len(NODULE_TYPE_VOCAB)
    + 2
    + 2 * len(CHARACTERISTIC_VOCAB)
    + 6
)


def _clean(value) -> str:
    return str(value or "").strip().lower()


def _normalise_name(value: str) -> str:
    value = value.replace("_", " ")
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return _clean(value)


def _parse_float(value: str) -> Optional[float]:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _first_text_fields(record: Dict) -> Dict:
    fields = record.get("text_fields", {})
    if isinstance(fields, list):
        return fields[0] if fields else {}
    return fields if isinstance(fields, dict) else {}


def _to_tensor(value) -> Optional[Tensor]:
    if value is None:
        return None
    try:
        return torch.as_tensor(value, dtype=torch.float64)
    except Exception:
        return None


def encode_patch_spatial_features(record: Optional[Dict]) -> List[float]:
    """Encode patch position as normalized center and extent in the original CT index space.

    Returns six values:
      center_x, center_y, center_z, extent_x, extent_y, extent_z.

    Center values are normalized to [-1, 1] over the original image extent.
    Extent values are normalized to [0, 1+] over the original image size.
    When metadata is unavailable, zeros are returned for center and ones for extent,
    corresponding to a whole-volume unknown fallback.
    """

    fallback = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    if record is None:
        return fallback

    image = record.get("image")
    meta = getattr(image, "meta", None)
    shape = getattr(image, "shape", None)
    if meta is None or shape is None or len(shape) < 4:
        return fallback

    affine = _to_tensor(meta.get("affine"))
    original_affine = _to_tensor(meta.get("original_affine"))
    spatial_shape = _to_tensor(meta.get("spatial_shape"))
    if affine is None or original_affine is None or spatial_shape is None or spatial_shape.numel() < 3:
        return fallback

    patch_shape = torch.as_tensor(tuple(int(x) for x in shape[-3:]), dtype=torch.float64)
    spatial_shape = spatial_shape[:3].to(dtype=torch.float64)
    spatial_denom = torch.clamp(spatial_shape - 1.0, min=1.0)

    # Current patch voxel corners in xyz index order. MONAI spatial tensors are
    # already ordered consistently with the affine used by MetaTensor.
    max_corner = torch.clamp(patch_shape - 1.0, min=0.0)
    corners = torch.stack(
        [
            torch.tensor([x, y, z, 1.0], dtype=torch.float64)
            for x in (0.0, float(max_corner[0]))
            for y in (0.0, float(max_corner[1]))
            for z in (0.0, float(max_corner[2]))
        ]
    )
    world_corners = (affine @ corners.T).T[:, :3]

    try:
        inv_original = torch.linalg.inv(original_affine)
    except RuntimeError:
        return fallback

    original_corners_h = torch.cat(
        [world_corners, torch.ones((world_corners.shape[0], 1), dtype=torch.float64)], dim=1
    )
    original_corners = (inv_original @ original_corners_h.T).T[:, :3]

    corner_min = torch.min(original_corners, dim=0).values
    corner_max = torch.max(original_corners, dim=0).values
    center = (corner_min + corner_max) / 2.0
    extent = torch.clamp(corner_max - corner_min + 1.0, min=1.0)

    center_norm = 2.0 * (center / spatial_denom) - 1.0
    extent_norm = extent / torch.clamp(spatial_shape, min=1.0)
    encoded = torch.cat([center_norm, extent_norm]).clamp(min=-2.0, max=2.0)
    return [float(x) for x in encoded.tolist()]


def encode_text_fields(text_fields: Dict, text_valid=1, patch_spatial_features: Optional[Sequence[float]] = None) -> List[float]:
    """Encode structured text_fields into a fixed dense vector.

    The vector is intentionally simple and deterministic: categorical values use
    one-hot or multi-hot slots; numeric values are normalized to a small range.
    """

    features: List[float] = []
    valid = float(text_valid)
    features.append(valid)

    diameter = _parse_float(text_fields.get("diameter_mm", ""))
    if diameter is None:
        features.extend([0.0, 1.0])
    else:
        features.extend([max(0.0, min(diameter / 30.0, 2.0)), 0.0])

    loc = _clean(text_fields.get("loc", ""))
    features.extend([1.0 if loc == item else 0.0 for item in LOC_VOCAB])

    position = _clean(text_fields.get("position", ""))
    position_tokens = set(position.replace("/", " ").split())
    features.extend([1.0 if item in position_tokens else 0.0 for item in POSITION_VOCAB])

    nodule_type = _clean(text_fields.get("nodule_type", ""))
    features.extend([1.0 if nodule_type == item else 0.0 for item in NODULE_TYPE_VOCAB])

    uncertainty = _clean(text_fields.get("uncertainty", ""))
    features.append(1.0 if "how many" in uncertainty else 0.0)
    features.append(1.0 if "is it" in uncertainty else 0.0)

    char_values = {name: 0.0 for name in CHARACTERISTIC_VOCAB}
    char_present = {name: 0.0 for name in CHARACTERISTIC_VOCAB}
    characteristics = str(text_fields.get("characteristics", "") or "")
    for match in re.finditer(r"([A-Za-z_ ]+?)\s+score\s+([-+]?\d+(?:\.\d+)?)", characteristics):
        name = _normalise_name(match.group(1))
        if name in char_values:
            char_values[name] = max(0.0, min(float(match.group(2)) / 6.0, 1.0))
            char_present[name] = 1.0
    features.extend([char_values[name] for name in CHARACTERISTIC_VOCAB])
    features.extend([char_present[name] for name in CHARACTERISTIC_VOCAB])
    features.extend(list(patch_spatial_features) if patch_spatial_features is not None else [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    if len(features) != TEXT_FEATURE_DIM:
        raise RuntimeError(f"Expected {TEXT_FEATURE_DIM} text features, got {len(features)}")
    return features


def batch_text_features(
    records: Sequence[Dict],
    device: Optional[torch.device | str] = None,
    non_blocking: bool = False,
    **kwargs,
) -> Tensor:
    encoded = []
    for record in records:
        text_valid = record.get("text_valid", 0)
        if isinstance(text_valid, (list, tuple)):
            text_valid = text_valid[0] if text_valid else 0
        encoded.append(
            encode_text_fields(
                _first_text_fields(record),
                text_valid=text_valid,
                patch_spatial_features=encode_patch_spatial_features(record),
            )
        )
    return torch.as_tensor(encoded, dtype=torch.float32).to(device=device, non_blocking=non_blocking, **kwargs)


class StructuredTextMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = TEXT_FEATURE_DIM,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, text_features: Tensor) -> Tensor:
        return self.net(text_features.float())


class TextConditionedRetinaNet(RetinaNet):
    """RetinaNet with FiLM conditioning on the classification branch only."""

    def __init__(
        self,
        spatial_dims: int,
        num_classes: int,
        num_anchors: int,
        feature_extractor: nn.Module,
        size_divisible: Sequence[int] | int = 1,
        use_list_output: bool = False,
        text_feature_dim: int = TEXT_FEATURE_DIM,
        text_embedding_dim: int = 128,
        text_hidden_dim: int = 128,
        text_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            num_classes=num_classes,
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
            use_list_output=use_list_output,
        )
        self.text_encoder = StructuredTextMLPEncoder(
            input_dim=text_feature_dim,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_hidden_dim,
            dropout=text_dropout,
        )
        self.text_to_film = nn.Linear(text_embedding_dim, 2 * self.feature_map_channels)
        nn.init.zeros_(self.text_to_film.weight)
        nn.init.zeros_(self.text_to_film.bias)
        self._text_features: Optional[Tensor] = None

    def set_text_features(self, text_features: Optional[Tensor]) -> None:
        self._text_features = text_features

    def _classification_head_forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        text_features = self._text_features
        if text_features is None:
            return self.classification_head(feature_maps)

        text_embedding = self.text_encoder(text_features)
        film = self.text_to_film(text_embedding)
        gamma, beta = torch.chunk(film, chunks=2, dim=1)

        cls_logits_maps = []
        for features in feature_maps:
            cls_features = self.classification_head.conv(features)
            view_shape = (features.shape[0], features.shape[1]) + (1,) * self.spatial_dims
            conditioned = cls_features * (1.0 + gamma.view(view_shape)) + beta.view(view_shape)
            cls_logits = self.classification_head.cls_logits(conditioned)
            cls_logits_maps.append(cls_logits)
        return cls_logits_maps

    def forward(self, images: Tensor) -> Dict[str, List[Tensor]] | List[Tensor]:
        features = self.feature_extractor(images)
        if isinstance(features, Tensor):
            feature_maps = [features]
        elif isinstance(features, dict):
            feature_maps = list(features.values())
        else:
            feature_maps = list(features)

        if not isinstance(feature_maps[0], Tensor):
            raise ValueError("feature_extractor output format must be Tensor, Dict[str, Tensor], or Sequence[Tensor].")

        cls_outputs = self._classification_head_forward(feature_maps)
        box_outputs = self.regression_head(feature_maps)
        if not self.use_list_output:
            return {self.cls_key: cls_outputs, self.box_reg_key: box_outputs}
        return cls_outputs + box_outputs


class TextConditionedRetinaNetDetector(RetinaNetDetector):
    def forward(
        self,
        input_images: List[Tensor] | Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        use_inferer: bool = False,
        text_features: Optional[Tensor] = None,
    ):
        if hasattr(self.network, "set_text_features"):
            self.network.set_text_features(text_features)
        try:
            return super().forward(input_images=input_images, targets=targets, use_inferer=use_inferer)
        finally:
            if hasattr(self.network, "set_text_features"):
                self.network.set_text_features(None)


def freeze_for_text_stage1(network: nn.Module) -> nn.Module:
    """Freeze image feature extractor and regression branch for first text-only stage."""

    for param in network.parameters():
        param.requires_grad = False

    for module_name in ("classification_head", "text_encoder", "text_to_film"):
        module = getattr(network, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
    return network
