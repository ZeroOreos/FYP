from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from Training.arcface import (
    ArcFaceModel,
    ArcMarginProduct,
    build_shared_class_subset,
    CosFaceMarginProduct,
    CurricularFaceMarginProduct,
    FaceBackbone,
    PartialFCArcMarginProduct,
    SubCenterArcMarginProduct,
)
from Training.config import RecognizerPolicy
from Utility.runtime import resolve_torch_device


class EmbeddingModel(Protocol):
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        ...


class TrainableRecognizer(Protocol):
    def forward_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        ...

    def forward_logits(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward_attack_outputs(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
        ...

    def predict_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        ...

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        ...

    def predict_eval_outputs(self, images: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
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
        logits = self.margin.training_outputs(embeddings, labels)["logits"]
        return logits, embeddings

    def predict_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.margin.inference_logits(embeddings)

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.forward_embeddings(images)
        return self.predict_logits_from_embeddings(embeddings)

    def predict_eval_outputs(self, images: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        embeddings = self.forward_embeddings(images)
        logits = self.predict_logits_from_embeddings(embeddings)
        return {
            "embeddings": embeddings,
            "predict_logits": logits,
            "member_logits": None,
        }

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor | None]:
        embeddings = self.forward_embeddings(images)
        margin_outputs = self.margin.training_outputs(embeddings, labels)
        return {
            "logits": margin_outputs["logits"],
            "loss_labels": margin_outputs["loss_labels"],
            "embeddings": embeddings,
            "predict_logits": margin_outputs["predict_logits"],
            "member_outputs": None,
        }

    def forward_attack_outputs(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor | None]:
        embeddings = self.forward_embeddings(images)
        margin_outputs = self.margin.training_outputs(embeddings, labels)
        return {
            "logits": margin_outputs["logits"],
            "loss_labels": margin_outputs["loss_labels"],
            "embeddings": embeddings,
            "predict_logits": margin_outputs["predict_logits"],
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


class ExternalRecognizerHead(nn.Module):
    def __init__(
        self,
        *,
        name: str,
        num_classes: int,
        embedding_dim: int,
        arcface_scale: float,
        arcface_margin: float,
        use_partial_fc: bool,
        partial_fc_negative_sample_rate: float,
        sub_center_count: int,
    ) -> None:
        super().__init__()
        lowered = name.strip().lower()
        if lowered == "arcface":
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
        elif lowered == "cosface":
            self.margin = CosFaceMarginProduct(
                embedding_dim,
                num_classes,
                s=arcface_scale,
                m=0.35,
            )
        elif lowered == "curricularface":
            self.margin = CurricularFaceMarginProduct(
                embedding_dim,
                num_classes,
                s=arcface_scale,
                m=arcface_margin,
            )
        else:
            raise ValueError(
                f"Unsupported recognizer ensemble member '{name}'. "
                "Supported values are 'arcface', 'cosface', and 'curricularface'."
            )
        self.name = lowered

    def training_outputs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        *,
        class_subset: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        margin_outputs = self.margin.training_outputs(embeddings, labels, class_subset=class_subset)
        loss_labels = margin_outputs.get("loss_labels", labels)
        loss_per_sample = F.cross_entropy(margin_outputs["logits"], loss_labels, reduction="none")
        return {
            "logits": margin_outputs["logits"],
            "loss_labels": loss_labels,
            "predict_logits": margin_outputs["predict_logits"],
            "loss_per_sample": loss_per_sample,
            "loss": loss_per_sample.mean(),
        }

    def inference_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.margin.inference_logits(embeddings)


class ExternalRecognizerEnsemble(nn.Module):
    def __init__(
        self,
        *,
        recognizers: list[RecognizerPolicy],
        num_classes: int,
        embedding_dim: int,
        arcface_scale: float,
        arcface_margin: float,
        use_partial_fc: bool,
        partial_fc_negative_sample_rate: float,
        sub_center_count: int,
        weight_strategy: str = "static",
    ) -> None:
        super().__init__()
        self.recognizer_specs = [policy for policy in recognizers if policy.enabled and policy.weight > 0]
        self.member_names = tuple(policy.name.strip().lower() for policy in self.recognizer_specs)
        self.member_weights = self._normalize_member_weights(self.recognizer_specs)
        self.use_partial_fc = bool(use_partial_fc)
        self.partial_fc_negative_sample_rate = float(partial_fc_negative_sample_rate)
        self.num_classes = int(num_classes)
        self.weight_strategy = str(weight_strategy).strip().lower()
        self.heads = nn.ModuleDict(
            {
                policy.name.strip().lower(): ExternalRecognizerHead(
                    name=policy.name,
                    num_classes=num_classes,
                    embedding_dim=embedding_dim,
                    arcface_scale=arcface_scale,
                    arcface_margin=arcface_margin,
                    use_partial_fc=use_partial_fc,
                    partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
                    sub_center_count=sub_center_count,
                )
                for policy in self.recognizer_specs
            }
        )

    @staticmethod
    def _normalize_member_weights(recognizers: list[RecognizerPolicy]) -> dict[str, float]:
        if not recognizers:
            return {}
        total = sum(max(0.0, float(policy.weight)) for policy in recognizers)
        if total <= 0:
            uniform = 1.0 / float(len(recognizers))
            return {policy.name.strip().lower(): uniform for policy in recognizers}
        return {
            policy.name.strip().lower(): max(0.0, float(policy.weight)) / total
            for policy in recognizers
        }

    def enabled(self) -> bool:
        return bool(self.recognizer_specs)

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if not self.recognizer_specs:
            return {}
        shared_subset = None
        if self.use_partial_fc:
            shared_subset = build_shared_class_subset(
                labels,
                out_features=self.num_classes,
                sample_rate=self.partial_fc_negative_sample_rate,
            )
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for name in self.member_names:
            head_outputs = self.heads[name].training_outputs(embeddings, labels, class_subset=shared_subset)
            outputs[name] = {
                "embeddings": embeddings,
                **head_outputs,
            }
        return outputs

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return self.forward_from_embeddings(embeddings, labels)

    def predict_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.recognizer_specs:
            raise RuntimeError("Recognizer ensemble is empty.")
        member_logits = {
            name: self.heads[name].inference_logits(embeddings)
            for name in self.member_names
        }
        return sum(self.member_weights[name] * member_logits[name] for name in self.member_names)

    def predict_eval_outputs_from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        if not self.recognizer_specs:
            raise RuntimeError("Recognizer ensemble is empty.")
        member_logits = {
            name: self.heads[name].inference_logits(embeddings)
            for name in self.member_names
        }
        pooled_logits = sum(self.member_weights[name] * member_logits[name] for name in self.member_names)
        return {
            "embeddings": embeddings,
            "predict_logits": pooled_logits,
            "member_logits": member_logits,
        }


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
        self.backbone = FaceBackbone(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dropout_p=dropout_p,
        )
        self.arcface_margin = (
            SubCenterArcMarginProduct(
                embedding_dim,
                num_classes,
                s=arcface_scale,
                m=arcface_margin,
                sub_center_count=sub_center_count,
            )
            if sub_center_count > 1
            else ArcMarginProduct(
                embedding_dim,
                num_classes,
                s=arcface_scale,
                m=arcface_margin,
            )
        )
        self.cosface_margin = CosFaceMarginProduct(
            embedding_dim,
            num_classes,
            s=arcface_scale,
            m=0.35,
        )
        self.curricularface_margin = CurricularFaceMarginProduct(
            embedding_dim,
            num_classes,
            s=arcface_scale,
            m=arcface_margin,
        )
        self.member_names = ("arcface", "cosface", "curricularface")
        self.embedding_dim = int(embedding_dim)
        self.model_name = "joint_pool"
        self.backbone_name = backbone_name
        self.sub_center_count = sub_center_count
        self.use_partial_fc = bool(use_partial_fc)
        self.partial_fc_negative_sample_rate = float(partial_fc_negative_sample_rate)
        self.num_classes = int(num_classes)
        self.member_weights = self._normalize_member_weights(member_weights)
        self.member_margins = {
            "arcface": self.arcface_margin,
            "cosface": self.cosface_margin,
            "curricularface": self.curricularface_margin,
        }

    def _normalize_member_weights(self, member_weights: dict[str, float] | None) -> dict[str, float]:
        provided = member_weights or {}
        weights = {name: float(provided.get(name, 0.0)) for name in self.member_names}
        if sum(max(0.0, value) for value in weights.values()) <= 0:
            weights = {"arcface": 0.4, "cosface": 0.3, "curricularface": 0.3}
        total = sum(max(0.0, value) for value in weights.values())
        return {name: max(0.0, value) / total for name, value in weights.items()}

    def _member_margins(self) -> dict[str, nn.Module]:
        return self.member_margins

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        target_size = 112 if self.backbone_name.startswith("iresnet") else 160
        resized = _resize(images, target_size)
        return (resized - 0.5) / 0.5

    def forward_member_outputs(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        embeddings = self.backbone(self.preprocess(images))
        shared_subset = None
        if self.use_partial_fc:
            shared_subset = build_shared_class_subset(
                labels,
                out_features=self.num_classes,
                sample_rate=self.partial_fc_negative_sample_rate,
            )
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for name, margin in self._member_margins().items():
            margin_outputs = margin.training_outputs(embeddings, labels, class_subset=shared_subset)
            loss_labels = margin_outputs.get("loss_labels", labels)
            loss_per_sample = F.cross_entropy(margin_outputs["logits"], loss_labels, reduction="none")
            outputs[name] = {
                "embeddings": embeddings,
                "logits": margin_outputs["logits"],
                "loss_labels": loss_labels,
                "predict_logits": margin_outputs["predict_logits"],
                "loss_per_sample": loss_per_sample,
                "loss": loss_per_sample.mean(),
            }
        return outputs

    def forward_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocess(images))

    def forward_logits(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        member_outputs = self.forward_member_outputs(images, labels)
        pooled_logits = sum(
            self.member_weights[name] * member_outputs[name]["logits"]
            for name in self.member_names
        )
        pooled_embeddings = member_outputs[self.member_names[0]]["embeddings"]
        return pooled_logits, pooled_embeddings

    def predict_member_logits_from_embeddings(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "arcface": self.arcface_margin.inference_logits(embeddings),
            "cosface": self.cosface_margin.inference_logits(embeddings),
            "curricularface": self.curricularface_margin.inference_logits(embeddings),
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

    def predict_eval_outputs(self, images: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        embeddings = self.forward_embeddings(images)
        member_logits = self.predict_member_logits_from_embeddings(embeddings)
        fused_logits = sum(self.member_weights[name] * member_logits[name] for name in self.member_names)
        return {
            "embeddings": embeddings,
            "predict_logits": fused_logits,
            "member_logits": member_logits,
        }

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
        member_outputs = self.forward_member_outputs(images, labels)
        pooled_loss_labels = member_outputs[self.member_names[0]]["loss_labels"]
        pooled_logits = sum(
            self.member_weights[name] * member_outputs[name]["logits"]
            for name in self.member_names
        )
        pooled_embeddings = member_outputs[self.member_names[0]]["embeddings"]
        pooled_predict_logits = sum(
            self.member_weights[name] * member_outputs[name]["predict_logits"]
            for name in self.member_names
        )
        return {
            "logits": pooled_logits,
            "loss_labels": pooled_loss_labels,
            "embeddings": pooled_embeddings,
            "predict_logits": pooled_predict_logits,
            "member_outputs": member_outputs,
        }

    def forward_attack_outputs(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
        embeddings = self.backbone(self.preprocess(images))
        shared_subset = None
        if self.use_partial_fc:
            shared_subset = build_shared_class_subset(
                labels,
                out_features=self.num_classes,
                sample_rate=self.partial_fc_negative_sample_rate,
            )
        member_logits: dict[str, torch.Tensor] = {}
        pooled_loss_labels = labels
        for name, margin in self._member_margins().items():
            margin_outputs = margin.training_outputs(embeddings, labels, class_subset=shared_subset)
            member_logits[name] = margin_outputs["logits"]
            if name == self.member_names[0]:
                pooled_loss_labels = margin_outputs.get("loss_labels", labels)
        pooled_logits = sum(
            self.member_weights[name] * member_logits[name]
            for name in self.member_names
        )
        return {
            "logits": pooled_logits,
            "loss_labels": pooled_loss_labels,
            "embeddings": embeddings,
            "predict_logits": pooled_logits,
            "member_outputs": None,
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
            member_weights=member_weights,
        )
    else:
        raise ValueError(
            f"Unsupported trainable target model '{model_name}'. "
            "Supported values are 'arcface', 'cosface', 'curricularface', and 'joint_pool'."
        )
    model = model.to(device)
    return model, device


def build_recognizer_ensemble(
    *,
    recognizer_specs: list[RecognizerPolicy],
    num_classes: int,
    embedding_dim: int,
    device: torch.device,
    arcface_scale: float,
    arcface_margin: float,
    use_partial_fc: bool,
    partial_fc_negative_sample_rate: float,
    sub_center_count: int,
    weight_strategy: str = "static",
) -> ExternalRecognizerEnsemble | None:
    enabled = [policy for policy in recognizer_specs if policy.enabled and policy.weight > 0]
    if not enabled:
        return None
    ensemble = ExternalRecognizerEnsemble(
        recognizers=enabled,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
        use_partial_fc=use_partial_fc,
        partial_fc_negative_sample_rate=partial_fc_negative_sample_rate,
        sub_center_count=sub_center_count,
        weight_strategy=weight_strategy,
    )
    return ensemble.to(device)


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
