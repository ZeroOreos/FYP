# ArcFaceModel(...) | ArcMarginProduct(...); forward(...) -> embeddings or logits
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceBackbone(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.embedding = nn.Linear(in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0, 1.0)

        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


class ArcFaceModel(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        self.backbone = FaceBackbone(embedding_dim=embedding_dim, pretrained=pretrained)

    def forward(self, x):
        return self.backbone(x)
