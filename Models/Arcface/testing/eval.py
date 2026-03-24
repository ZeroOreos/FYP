# python3 eval.py [...] -> evaluation output
import torch
import torch.nn.functional as F
from typing import cast
from PIL import Image
from torchvision import transforms
from Arcface.testing.model import ArcFaceModel


def load_image(path, img_size=112):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = Image.open(path).convert("RGB")
    tensor_img = cast(torch.Tensor, tf(img))
    return tensor_img.unsqueeze(0)


def cosine_compare(model, img1_path, img2_path, device):
    img1 = load_image(img1_path).to(device)
    img2 = load_image(img2_path).to(device)

    model.eval()
    with torch.no_grad():
        emb1 = model.backbone(img1)
        emb2 = model.backbone(img2)
        sim = F.cosine_similarity(emb1, emb2)

    return sim.item()


def main():
    checkpoint = torch.load("best_arcface.pth", map_location="cpu")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ArcFaceModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    img1 = "test/person1_1.jpg"
    img2 = "test/person1_2.jpg"

    score = cosine_compare(model, img1, img2, device)
    print(f"Cosine similarity: {score:.4f}")

    threshold = 0.5
    if score > threshold:
        print("Same identity")
    else:
        print("Different identity")


if __name__ == "__main__":
    main()
