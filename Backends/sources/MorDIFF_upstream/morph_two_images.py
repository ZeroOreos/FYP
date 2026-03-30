import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIFFAE_DIR = SCRIPT_DIR.parent / "dependencies" / "diffae"
path_to_diff_model = os.environ.get("FYP_DIFFAE_DIR", str(DEFAULT_DIFFAE_DIR))

sys.path.append(path_to_diff_model)

from templates import *
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torch.functional import F
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    mps_backend = getattr(torch.backends, "mps", None)
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine")
        return "cuda:0"
    if requested in {"mps", "coreml"}:
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS/CoreML-style acceleration was requested but is not available on this machine")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device selection: {requested}")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--img1",
                        type=str,
                        help="image1 path")
    parser.add_argument("--img2",
                        type=str,
                        help="image2 path")
    parser.add_argument("--output",
                        type=str,
                        help="output path")
    parser.add_argument("--stochastic-steps",
                        type=int,
                        default=250,
                        help="number of DDIM stochastic encoding steps")
    parser.add_argument("--render-steps",
                        type=int,
                        default=20,
                        help="number of rendering steps")
    parser.add_argument("--save-midpoint-only",
                        action="store_true",
                        help="save only the alpha=0.5 morph image instead of a comparison grid")
    parser.add_argument("--device",
                        type=str,
                        default="auto",
                        choices=["auto", "cuda", "mps", "coreml", "cpu"],
                        help="device preference order: auto prefers CUDA, then MPS, then CPU")

    args = parser.parse_args()

    # load the model
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    conf = ffhq256_autoenc()
    # print(conf.name)
    model = LitModel(conf)
    checkpoint_path = Path(path_to_diff_model) / "checkpoints" / conf.name / "last.ckpt"
    state = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # images to fuse: needs to be aligned
    image1 = args.img1
    image2 = args.img2
    outfolder = args.output

    image_size = conf.img_size # 256

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    to_pil_image = transforms.ToPILImage()

    def load_image(path):
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('RGB')

        assert img.size[0] == img.size[1] == 256

        if transform is not None:
            img = transform(img)

        return img

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    img1 = load_image(image1)
    img2 = load_image(image2)

    batch = torch.stack([
        img1, 
        img2
    ])

    cond = model.encode(batch.to(device))

    T = args.stochastic_steps
    xT = model.encode_stochastic(batch.to(device), cond, T=T)

    if args.save_midpoint_only:
        alpha = torch.tensor([0.5]).to(cond.device)
    else:
        alpha = torch.tensor([0.0, 0.5, 1.0]).to(cond.device)
    intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

    def cos(a, b):
        a = a.view(-1)
        b = b.view(-1)
        a = F.normalize(a, dim=0)
        b = F.normalize(b, dim=0)
        return (a * b).sum()

    theta = torch.arccos(cos(xT[0], xT[1]))
    x_shape = xT[0].shape
    intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
    intp_x = intp_x.view(-1, *x_shape)

    pred = model.render(intp_x, intp, T=args.render_steps)

    name1 = image1.split(".")[0].split("/")[-1]
    name2 = image2.split(".")[0].split("/")[-1]
    if args.save_midpoint_only:
        midpoint = pred[0].detach().cpu().clamp(0, 1)
        midpoint_image = to_pil_image(midpoint)
        midpoint_image.save(os.path.join(outfolder, f"morph_{name1}_and_{name2}.png"))
    else:
        # torch.manual_seed(1)
        #fig, ax = plt.subplots(1, 10, figsize=(5*10, 5))
        fig, ax = plt.subplots(1, 3, figsize=(5*3, 5))
        for i in range(len(alpha)):
            ax[i].imshow(pred[i].permute(1, 2, 0).cpu())

        plt.savefig(os.path.join(outfolder, f"comparison_{name1}_and_{name2}.png"))
        plt.close("all")
        pred_image = to_pil_image(pred[1].detach().cpu().clamp(0, 1))
        pred_image.save(os.path.join(outfolder, f"morphed_{name1}_and_{name2}.png"))
