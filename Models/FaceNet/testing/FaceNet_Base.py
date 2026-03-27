# python3 FaceNet_Base.py -> embedding utility output
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from Utility.runtime import resolve_torch_device

device = torch.device(resolve_torch_device())

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=True,
    device=device
)

model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        raise ValueError(f"No face detected in {image_path}")

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face)

    return embedding.squeeze(0)

def cosine_score(image1, image2):
    emb1 = get_embedding(image1)
    emb2 = get_embedding(image2)

    score = F.cosine_similarity(
        emb1.unsqueeze(0),
        emb2.unsqueeze(0)
    ).item()

    return score

if __name__ == "__main__":
    img1 = "Images/test0.jpeg"
    img2 = "Images/test0.jpeg"
    
    score = cosine_score(img1, img2)
    threshold = 0.75
    pred = "same person" if score >= threshold else "different person"

    print(f"Cosine similarity: {score:.4f}")
    print(f"Prediction: {pred}")
