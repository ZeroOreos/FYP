from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models


@dataclass(frozen=True)
class ArcFaceCeilingSpec:
    detector_name: str = "RetinaFace-class"
    alignment_points: int = 5
    normalized_crop_size: int = 112
    dataset_name: str = "WebFace42M"
    target_backbone: str = "ir_resnet200"
    target_embedding_dim: int = 512
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5
    partial_fc_negative_sample_rate: float = 0.3
    lr_start: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    published_world_size: int = 32
    published_batch_size_per_gpu: int = 128
    sub_center_count: int = 1


@dataclass(frozen=True)
class SharedClassSubset:
    class_indices: torch.Tensor
    remapped_labels: torch.Tensor


def build_shared_class_subset(
    labels: torch.Tensor,
    *,
    out_features: int,
    sample_rate: float,
) -> SharedClassSubset | None:
    if sample_rate >= 1.0 or out_features <= 1:
        return None
    unique_labels = torch.unique(labels.detach()).to(torch.long)
    all_indices = torch.arange(out_features, device=labels.device, dtype=torch.long)
    negative_mask = torch.ones(out_features, device=labels.device, dtype=torch.bool)
    negative_mask[unique_labels] = False
    negative_indices = all_indices[negative_mask]

    if negative_indices.numel() > 0:
        negative_count = max(1, int(round(negative_indices.numel() * float(sample_rate))))
        permutation = torch.randperm(negative_indices.numel(), device=labels.device)
        sampled_negatives = negative_indices[permutation[:negative_count]]
        sampled_indices = torch.cat([unique_labels, sampled_negatives], dim=0)
    else:
        sampled_indices = unique_labels

    label_matches = sampled_indices.unsqueeze(0) == labels.unsqueeze(1)
    if not torch.all(label_matches.any(dim=1)):
        raise RuntimeError("Shared class subset failed to preserve all positive class centers.")
    remapped_labels = label_matches.to(torch.long).argmax(dim=1)
    return SharedClassSubset(class_indices=sampled_indices, remapped_labels=remapped_labels)


class ArcFaceHead(nn.Module):
    """BN-Dropout-FC-BN head."""

    def __init__(self, in_features: int, embedding_dim: int, dropout_p: float = 0.4) -> None:
        super().__init__()
        self.pre_bn = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(in_features, embedding_dim)
        self.post_bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.post_bn(x)
        return F.normalize(x, p=2, dim=1)


class FaceBackbone(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_gradient_checkpointing: bool = False,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.backbone_name = backbone_name.strip().lower()

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            self.head = ArcFaceHead(base.fc.in_features, embedding_dim, dropout_p=dropout_p)
            self._forward_impl = self._forward_torchvision
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.resnet50(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            self.head = ArcFaceHead(base.fc.in_features, embedding_dim, dropout_p=dropout_p)
            self._forward_impl = self._forward_torchvision
        elif self.backbone_name == "iresnet100":
            self.iresnet = iresnet100(
                pretrained=False,
                num_features=embedding_dim,
                dropout=dropout_p,
                fp16=False,
            )
            self._forward_impl = self._forward_iresnet
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                "Supported values are 'resnet18', 'resnet50', and 'iresnet100'."
            )

    def _forward_stage(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _forward_torchvision(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_stage(self.stem, x)
        x = self._forward_stage(self.layer1, x)
        x = self._forward_stage(self.layer2, x)
        x = self._forward_stage(self.layer3, x)
        x = self._forward_stage(self.layer4, x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def _forward_iresnet(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            previous = using_ckpt
            try:
                globals()["using_ckpt"] = True
                return self.iresnet(x)
            finally:
                globals()["using_ckpt"] = previous
        return self.iresnet(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.50,
        sub_center_count: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub_center_count = max(1, int(sub_center_count))

        self.weight = nn.Parameter(torch.empty(out_features * self.sub_center_count, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _weight_rows_for_classes(self, class_indices: torch.Tensor) -> torch.Tensor:
        if self.sub_center_count == 1:
            return class_indices
        offsets = torch.arange(self.sub_center_count, device=class_indices.device, dtype=torch.long)
        return (class_indices.unsqueeze(1) * self.sub_center_count + offsets.unsqueeze(0)).reshape(-1)

    def _cosine_logits(
        self,
        embeddings: torch.Tensor,
        weights: torch.Tensor,
        class_count: int,
    ) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(weights))
        if self.sub_center_count > 1:
            cosine = cosine.view(embeddings.size(0), class_count, self.sub_center_count).max(dim=2).values
        return cosine

    def inference_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self._cosine_logits(embeddings, self.weight, self.out_features) * self.s

    def _arc_logits(
        self,
        embeddings: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
        class_count: int,
    ) -> torch.Tensor:
        cosine = self._cosine_logits(embeddings, weights, class_count)
        cosine = cosine.clamp(-1.0, 1.0)

        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._arc_logits(embeddings, self.weight, labels, self.out_features)

    def training_outputs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_subset: SharedClassSubset | None = None,
    ) -> dict[str, torch.Tensor]:
        if class_subset is None:
            logits = self.forward(embeddings, labels)
            loss_labels = labels
        else:
            sampled_weights = self.weight[self._weight_rows_for_classes(class_subset.class_indices)]
            logits = self._arc_logits(
                embeddings,
                sampled_weights,
                class_subset.remapped_labels,
                class_subset.class_indices.numel(),
            )
            loss_labels = class_subset.remapped_labels
        return {
            "logits": logits,
            "loss_labels": loss_labels,
            "predict_logits": logits,
        }


class SubCenterArcMarginProduct(ArcMarginProduct):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.50,
        sub_center_count: int = 3,
    ) -> None:
        if sub_center_count < 2:
            raise ValueError("Sub-center ArcFace requires sub_center_count >= 2.")
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            s=s,
            m=m,
            sub_center_count=sub_center_count,
        )


class PartialFCArcMarginProduct(ArcMarginProduct):
    """Practical Partial FC approximation with optional sub-centers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.50,
        negative_sample_rate: float = 0.3,
        sub_center_count: int = 1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            s=s,
            m=m,
            sub_center_count=sub_center_count,
        )
        self.negative_sample_rate = float(negative_sample_rate)

    def _sampled_outputs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_subset: SharedClassSubset | None = None,
    ) -> dict[str, torch.Tensor]:
        if class_subset is not None:
            sampled_indices = class_subset.class_indices
            remapped_labels = class_subset.remapped_labels
            sampled_weights = self.weight[self._weight_rows_for_classes(sampled_indices)]
            sampled_logits = self._arc_logits(embeddings, sampled_weights, remapped_labels, sampled_indices.numel())
            return {
                "logits": sampled_logits,
                "loss_labels": remapped_labels,
                "predict_logits": sampled_logits,
            }
        if (not self.training) or self.negative_sample_rate >= 1.0 or self.out_features <= 1:
            logits = super().forward(embeddings, labels)
            return {
                "logits": logits,
                "loss_labels": labels,
                "predict_logits": logits,
            }

        unique_labels = torch.unique(labels.detach()).to(torch.long)
        all_indices = torch.arange(self.out_features, device=labels.device, dtype=torch.long)
        negative_mask = torch.ones(self.out_features, device=labels.device, dtype=torch.bool)
        negative_mask[unique_labels] = False
        negative_indices = all_indices[negative_mask]

        if negative_indices.numel() > 0:
            negative_count = max(1, int(round(negative_indices.numel() * self.negative_sample_rate)))
            permutation = torch.randperm(negative_indices.numel(), device=labels.device)
            sampled_negatives = negative_indices[permutation[:negative_count]]
            sampled_indices = torch.cat([unique_labels, sampled_negatives], dim=0)
        else:
            sampled_indices = unique_labels

        sampled_weights = self.weight[self._weight_rows_for_classes(sampled_indices)]
        label_matches = sampled_indices.unsqueeze(0) == labels.unsqueeze(1)
        if not torch.all(label_matches.any(dim=1)):
            raise RuntimeError("Partial FC failed to preserve all positive class centers in the sampled subset.")
        remapped_labels = label_matches.to(torch.long).argmax(dim=1)
        sampled_logits = self._arc_logits(embeddings, sampled_weights, remapped_labels, sampled_indices.numel())
        return {
            "logits": sampled_logits,
            "loss_labels": remapped_labels,
            "predict_logits": sampled_logits,
        }

    def training_outputs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_subset: SharedClassSubset | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._sampled_outputs(embeddings, labels, class_subset=class_subset)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.training_outputs(embeddings, labels)["logits"]


class CosFaceMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return self.s * (cosine - one_hot * self.m)

    def inference_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine * self.s

    def training_outputs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_subset: SharedClassSubset | None = None,
    ) -> dict[str, torch.Tensor]:
        if class_subset is None:
            logits = self.forward(embeddings, labels)
            loss_labels = labels
        else:
            sampled_weights = self.weight[class_subset.class_indices]
            cosine = F.linear(F.normalize(embeddings), F.normalize(sampled_weights))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, class_subset.remapped_labels.view(-1, 1), 1.0)
            logits = self.s * (cosine - one_hot * self.m)
            loss_labels = class_subset.remapped_labels
        return {
            "logits": logits,
            "loss_labels": loss_labels,
            "predict_logits": logits,
        }


class CurricularFaceMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.50,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("t", torch.zeros(1))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0, 1.0)
        target_logit = cosine[torch.arange(0, embeddings.size(0), device=embeddings.device), labels].view(-1, 1)
        sine = torch.sqrt(torch.clamp(1.0 - target_logit.pow(2), min=1e-9))
        cosine_with_margin = target_logit * self.cos_m - sine * self.sin_m
        hard_mask = cosine > cosine_with_margin
        final_target = torch.where(target_logit > self.threshold, cosine_with_margin, target_logit - self.mm)
        hard_examples = cosine[hard_mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1.0 - 0.01) * self.t
        t_value = self.t.to(dtype=hard_examples.dtype, device=hard_examples.device)
        cosine[hard_mask] = hard_examples * (t_value + hard_examples)
        cosine.scatter_(1, labels.view(-1, 1).long(), final_target)
        return cosine * self.s

    def inference_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine.clamp(-1.0, 1.0) * self.s

    def training_outputs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_subset: SharedClassSubset | None = None,
    ) -> dict[str, torch.Tensor]:
        if class_subset is None:
            logits = self.forward(embeddings, labels)
            loss_labels = labels
        else:
            sampled_weights = self.weight[class_subset.class_indices]
            cosine = F.linear(F.normalize(embeddings), F.normalize(sampled_weights))
            cosine = cosine.clamp(-1.0, 1.0)
            target_logit = cosine[
                torch.arange(0, embeddings.size(0), device=embeddings.device),
                class_subset.remapped_labels,
            ].view(-1, 1)
            sine = torch.sqrt(torch.clamp(1.0 - target_logit.pow(2), min=1e-9))
            cosine_with_margin = target_logit * self.cos_m - sine * self.sin_m
            hard_mask = cosine > cosine_with_margin
            final_target = torch.where(target_logit > self.threshold, cosine_with_margin, target_logit - self.mm)
            final_target = final_target.to(dtype=cosine.dtype, device=cosine.device)
            hard_examples = cosine[hard_mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1.0 - 0.01) * self.t
            t_value = self.t.to(dtype=hard_examples.dtype, device=hard_examples.device)
            cosine[hard_mask] = hard_examples * (t_value + hard_examples)
            cosine.scatter_(1, class_subset.remapped_labels.view(-1, 1).long(), final_target)
            logits = cosine * self.s
            loss_labels = class_subset.remapped_labels
        if class_subset is None:
            logits = self.forward(embeddings, labels)
            loss_labels = labels
        return {
            "logits": logits,
            "loss_labels": loss_labels,
            "predict_logits": logits,
        }


class ArcFaceModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_gradient_checkpointing: bool = False,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__()
        self.backbone = FaceBackbone(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


using_ckpt = False


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x, use_reentrant=False)
        return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(
        self,
        block,
        layers,
        dropout=0,
        num_features=512,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        fp16=False,
    ):
        super().__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        autocast_enabled = self.fp16 and torch.cuda.is_available() and x.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if autocast_enabled else x)
        x = self.features(x)
        return F.normalize(x, p=2, dim=1)


def iresnet100(pretrained=False, progress=True, **kwargs):
    del progress
    if pretrained:
        raise ValueError("Pretrained iresnet100 weights are not packaged in this repo path.")
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)
