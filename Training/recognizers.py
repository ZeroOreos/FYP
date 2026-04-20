from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from Training.arcface import (
    ArcFaceModel,
    ArcMarginProduct,
    CosFaceMarginProduct,
    CurricularFaceMarginProduct,
    PartialFCArcMarginProduct,
    SubCenterArcMarginProduct,
)
from Utility.runtime import resolve_torch_device


class EmbeddingModel(Protocol):
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        ...


class TrainableRecognizer(Protocol):
    def forward_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        ...

    def forward_logits(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def predict_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        ...

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        ...


def _resize(images: torch.Tensor, size: int) -> torch.Tensor:
    if images.shape[-1] == size and images.shape[-2] == size:
        return images
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


@dataclass
class SurrogateWrapper:
    name: str
    module: nn.Module

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MarginTarget(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        embedding_dim: int,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_gradient_checkpointing: bool = False,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
        use_partial_fc: bool = True,
        partial_fc_negative_sample_rate: float = 0.3,
        sub_center_count: int = 1,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__()
        lowered_name = model_name.strip().lower()
        if lowered_name not in {"arcface", "cosface", "curricularface"}:
            raise ValueError(f"Unsupported margin target model: {model_name}")
        self.backbone = ArcFaceModel(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dropout_p=dropout_p,
        )
        if lowered_name == "arcface":
            if use_partial_fc:
                self.margin = PartialFCArcMarginProduct(
                    embedding_dim,
                    num_classes,
                    s=arcface_scale,
                    m=arcface_margin,
                    negative_sample_rate=partial_fc_negative_sample_rate,
                    sub_center_count=sub_center_count,
                )
            else:
                if sub_center_count > 1:
                    self.margin = SubCenterArcMarginProduct(
                        embedding_dim,
                        num_classes,
                        s=arcface_scale,
                        m=arcface_margin,
                        sub_center_count=sub_center_count,
                    )
                else:
                    self.margin = ArcMarginProduct(
                        embedding_dim,
                        num_classes,
                        s=arcface_scale,
                        m=arcface_margin,
                    )
        elif lowered_name == "cosface":
            self.margin = CosFaceMarginProduct(
                embedding_dim,
                num_classes,
                s=arcface_scale,
                m=arcface_margin,
            )
        else:
            self.margin = CurricularFaceMarginProduct(
                embedding_dim,
                num_classes,
                s=arcface_scale,
                m=arcface_margin,
            )
        self.model_name = lowered_name
        self.backbone_name = backbone_name
        self.sub_center_count = sub_center_count

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        target_size = 112 if self.backbone_name.startswith("iresnet") else 160
        resized = _resize(images, target_size)
        return (resized - 0.5) / 0.5

    def forward_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocess(images))

    def forward_logits(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.forward_embeddings(images)
        logits = self.margin(embeddings, labels)
        return logits, embeddings

    def predict_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.margin.inference_logits(embeddings)

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.forward_embeddings(images)
        return self.predict_logits_from_embeddings(embeddings)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor | None]:
        embeddings = self.forward_embeddings(images)
        logits = self.margin(embeddings, labels)
        predict_logits = self.predict_logits_from_embeddings(embeddings)
        return {
            "logits": logits,
            "embeddings": embeddings,
            "predict_logits": predict_logits,
            "member_outputs": None,
        }


class ArcFaceTarget(MarginTarget):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_gradient_checkpointing: bool = False,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
        use_partial_fc: bool = True,
        partial_fc_negative_sample_rate: float = 0.3,
        sub_center_count: int = 1,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__(
            model_name="arcface",
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=use_partial_fc,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=sub_center_count,
            dropout_p=dropout_p,
        )


class CosFaceTarget(MarginTarget):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_gradient_checkpointing: bool = False,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.35,
        use_partial_fc: bool = True,
        partial_fc_negative_sample_rate: float = 0.3,
        sub_center_count: int = 1,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__(
            model_name="cosface",
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=use_partial_fc,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=sub_center_count,
            dropout_p=dropout_p,
        )


class CurricularFaceTarget(MarginTarget):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_gradient_checkpointing: bool = False,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
        use_partial_fc: bool = True,
        partial_fc_negative_sample_rate: float = 0.3,
        sub_center_count: int = 1,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__(
            model_name="curricularface",
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=use_partial_fc,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=sub_center_count,
            dropout_p=dropout_p,
        )


class TargetSelfSurrogate(SurrogateWrapper):
    def __init__(self, target: TrainableRecognizer) -> None:
        super().__init__(name="target", module=target)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.module.forward_embeddings(images)


class JointRecognizerPool(nn.Module):
    """Joint defended pool with ArcFace, CosFace, and CurricularFace members."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_gradient_checkpointing: bool = False,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
        use_partial_fc: bool = True,
        partial_fc_negative_sample_rate: float = 0.3,
        sub_center_count: int = 1,
        dropout_p: float = 0.4,
        member_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.arcface = ArcFaceTarget(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=use_partial_fc,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=sub_center_count,
            dropout_p=dropout_p,
        )
        self.cosface = CosFaceTarget(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=0.35,
            use_partial_fc=False,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=1,
            dropout_p=dropout_p,
        )
        self.curricularface = CurricularFaceTarget(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=False,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=1,
            dropout_p=dropout_p,
        )
        self.member_names = ("arcface", "cosface", "curricularface")
        self.embedding_dim = int(embedding_dim)
        self.model_name = "joint_pool"
        self.backbone_name = backbone_name
        self.sub_center_count = sub_center_count
        self.member_weights = self._normalize_member_weights(member_weights)

    def _normalize_member_weights(self, member_weights: dict[str, float] | None) -> dict[str, float]:
        provided = member_weights or {}
        weights = {name: float(provided.get(name, 0.0)) for name in self.member_names}
        if sum(max(0.0, value) for value in weights.values()) <= 0:
            weights = {"arcface": 0.4, "cosface": 0.3, "curricularface": 0.3}
        total = sum(max(0.0, value) for value in weights.values())
        return {name: max(0.0, value) / total for name, value in weights.items()}

    def _member_modules(self) -> dict[str, MarginTarget]:
        return {
            "arcface": self.arcface,
            "cosface": self.cosface,
            "curricularface": self.curricularface,
        }

    def _component_embeddings(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.arcface.forward_embeddings(images),
            self.cosface.forward_embeddings(images),
            self.curricularface.forward_embeddings(images),
        )

    def forward_member_outputs(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for name, member in self._member_modules().items():
            embeddings = member.forward_embeddings(images)
            logits = member.margin(embeddings, labels)
            predict_logits = member.predict_logits_from_embeddings(embeddings)
            outputs[name] = {
                "embeddings": embeddings,
                "logits": logits,
                "predict_logits": predict_logits,
            }
        return outputs

    def forward_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self._component_embeddings(images)
        return torch.cat(embeddings, dim=1)

    def forward_logits(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        member_outputs = self.forward_member_outputs(images, labels)
        pooled_logits = sum(
            self.member_weights[name] * member_outputs[name]["logits"]
            for name in self.member_names
        )
        pooled_embeddings = torch.cat(
            [member_outputs[name]["embeddings"] for name in self.member_names],
            dim=1,
        )
        return pooled_logits, pooled_embeddings

    def predict_member_logits_from_embeddings(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        chunks = torch.split(embeddings, self.embedding_dim, dim=1)
        if len(chunks) != 3:
            raise ValueError(
                f"Joint pool embeddings must split into exactly three chunks of size {self.embedding_dim}; "
                f"got shape {tuple(embeddings.shape)}"
            )
        return {
            "arcface": self.arcface.predict_logits_from_embeddings(chunks[0]),
            "cosface": self.cosface.predict_logits_from_embeddings(chunks[1]),
            "curricularface": self.curricularface.predict_logits_from_embeddings(chunks[2]),
        }

    def predict_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        member_logits = self.predict_member_logits_from_embeddings(embeddings)
        return sum(self.member_weights[name] * member_logits[name] for name in self.member_names)

    def predict_member_logits(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.forward_embeddings(images)
        return self.predict_member_logits_from_embeddings(embeddings)

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.forward_embeddings(images)
        return self.predict_logits_from_embeddings(embeddings)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
        member_outputs = self.forward_member_outputs(images, labels)
        pooled_logits = sum(
            self.member_weights[name] * member_outputs[name]["logits"]
            for name in self.member_names
        )
        pooled_embeddings = torch.cat(
            [member_outputs[name]["embeddings"] for name in self.member_names],
            dim=1,
        )
        pooled_predict_logits = sum(
            self.member_weights[name] * member_outputs[name]["predict_logits"]
            for name in self.member_names
        )
        return {
            "logits": pooled_logits,
            "embeddings": pooled_embeddings,
            "predict_logits": pooled_predict_logits,
            "member_outputs": member_outputs,
        }


def build_target_model(
    *,
    model_name: str,
    num_classes: int,
    embedding_dim: int,
    device_name: str,
    backbone_name: str = "resnet18",
    use_gradient_checkpointing: bool = False,
    arcface_scale: float = 64.0,
    arcface_margin: float = 0.5,
    use_partial_fc: bool = True,
    partial_fc_negative_sample_rate: float = 0.3,
    sub_center_count: int = 1,
    dropout_p: float = 0.4,
    member_weights: dict[str, float] | None = None,
) -> tuple[nn.Module, torch.device]:
    # All supported trainable targets share the same margin-target interface.
    device = torch.device(resolve_torch_device(device_name))
    lowered = model_name.strip().lower()
    if lowered == "arcface":
        model: MarginTarget = ArcFaceTarget(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=True,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=use_partial_fc,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=sub_center_count,
            dropout_p=dropout_p,
        )
    elif lowered == "cosface":
        model = CosFaceTarget(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=True,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=use_partial_fc,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=sub_center_count,
            dropout_p=dropout_p,
        )
    elif lowered == "curricularface":
        model = CurricularFaceTarget(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=True,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=use_partial_fc,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=sub_center_count,
            dropout_p=dropout_p,
        )
    elif lowered in {"joint_pool", "jointpool", "pool"}:
        model = JointRecognizerPool(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=True,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            use_partial_fc=use_partial_fc,
            partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
            sub_center_count=sub_center_count,
            dropout_p=dropout_p,
        )
    else:
        raise ValueError(
            f"Unsupported trainable target model '{model_name}'. "
            "Supported values are 'arcface', 'cosface', 'curricularface', and 'joint_pool'."
        )
    model = model.to(device)
    return model, device


def build_surrogates(
    surrogate_names: list[str],
    target_model: TrainableRecognizer,
    device: torch.device,
) -> dict[str, SurrogateWrapper]:
    del device
    surrogates: dict[str, SurrogateWrapper] = {}
    for name in surrogate_names:
        lowered = name.lower()
        if lowered == "target":
            surrogates["target"] = TargetSelfSurrogate(target_model)
            continue
        raise ValueError(
            f"Unsupported recognizer-side surrogate '{name}'. "
            "The refactored ensemble code only supports 'target' here; external attack pressure now belongs on the attacker side of the ensemble."
        )
    return surrogates
