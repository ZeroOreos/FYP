# python3 insightFace_embed_extrac.py -> embedding output
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

from insightface.model_zoo import get_model

from Utility.runtime import resolve_onnx_providers


MAIN_DIR = "Dataset/CelebA/main"
OUTPUT_FILE = "main_antelopev2_embeddings.npz"

MODEL_ROOT = os.path.expanduser("~/.insightface/models")

MODEL_PACK = "antelopev2"

PROVIDERS = resolve_onnx_providers()

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_recognition_model(model_root, pack_name):
    """
    Finds the ArcFace recognition ONNX inside the antelope pack.
    We skip detector/gender/etc. and choose the recognition backbone.
    """
    pack_dir = os.path.join(model_root, pack_name)
    if not os.path.isdir(pack_dir):
        raise FileNotFoundError(
            f"Model pack folder not found: {pack_dir}\n"
            f"Make sure '{pack_name}' is downloaded under {model_root}"
        )

    onnx_files = sorted(glob.glob(os.path.join(pack_dir, "*.onnx")))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files found in {pack_dir}")

    preferred_keywords = [
        "glintr100",
        "webface_r50",
        "r100",
        "r50",
    ]

    for kw in preferred_keywords:
        for f in onnx_files:
            if kw in os.path.basename(f).lower():
                return f

    skip_keywords = [
        "det",
        "detect",
        "scrfd",
        "retina",
        "landmark",
        "2d106",
        "genderage",
    ]

    candidates = []
    for f in onnx_files:
        name = os.path.basename(f).lower()
        if not any(sk in name for sk in skip_keywords):
            candidates.append(f)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print("[WARN] Multiple recognition models found:")
        for c in candidates:
            print("   ", c)
        print("[WARN] Using first match.")
        return candidates[0]

    raise RuntimeError(
        f"Could not identify recognition model in {pack_dir}.\n"
        f"Found files: {[os.path.basename(x) for x in onnx_files]}"
    )


def scan_partitioned_folder(root_dir):
    """
    Returns:
      image_paths: list[str]
      labels: np.ndarray[int]
      identities: list[str]
      class_names: list[str]
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Main directory not found: {root_dir}")

    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    if not class_names:
        raise ValueError(f"No identity folders found in: {root_dir}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    image_paths = []
    labels = []
    identities = []

    for identity in class_names:
        identity_dir = os.path.join(root_dir, identity)

        files = sorted(os.listdir(identity_dir))
        for fname in files:
            fpath = os.path.join(identity_dir, fname)
            ext = os.path.splitext(fname)[1].lower()

            if os.path.isfile(fpath) and ext in VALID_EXTS:
                image_paths.append(fpath)
                labels.append(class_to_idx[identity])
                identities.append(identity)

    if not image_paths:
        raise ValueError(f"No images found under: {root_dir}")

    return image_paths, np.array(labels, dtype=np.int64), identities, class_names


def load_recognition_model():
    rec_model_path = find_recognition_model(MODEL_ROOT, MODEL_PACK)
    print(f"[INFO] Recognition model: {rec_model_path}")
    print(f"[INFO] Providers requested: {PROVIDERS}")

    model = get_model(rec_model_path, providers=PROVIDERS)

    print("[INFO] Model loaded successfully.")
    return model


def preprocess_for_arcface(img_bgr, input_size=(112, 112)):
    """
    ArcFace recognition models usually expect aligned face crops.
    If your CelebA images are already aligned/cropped, simple resize is fine.
    """
    if img_bgr is None:
        return None

    resized = cv2.resize(img_bgr, input_size)
    return resized


def extract_embeddings(model, image_paths):
    embeddings = []
    kept_paths = []
    failed_paths = []

    input_size = (112, 112)
    if hasattr(model, "input_shape") and model.input_shape is not None:
        try:
            h = int(model.input_shape[2])
            w = int(model.input_shape[3])
            input_size = (w, h)
        except Exception:
            pass

    print(f"[INFO] Using recognition input size: {input_size}")

    for path in tqdm(image_paths, desc="Extracting embeddings"):
        img = cv2.imread(path)
        if img is None:
            failed_paths.append(path)
            continue

        img = preprocess_for_arcface(img, input_size=input_size)

        try:
            feat = model.get_feat(img)
            feat = np.asarray(feat).reshape(-1)

            embeddings.append(feat.astype(np.float32))
            kept_paths.append(path)

        except Exception as e:
            print(f"[WARN] Failed: {path}: {e}")
            failed_paths.append(path)

    if not embeddings:
        raise RuntimeError("No embeddings were extracted.")

    embeddings = np.stack(embeddings, axis=0)
    return embeddings, kept_paths, failed_paths


def main():
    print(f"[INFO] Scanning folder: {MAIN_DIR}")
    image_paths, labels, identities, class_names = scan_partitioned_folder(MAIN_DIR)

    print(f"[INFO] Identities: {len(class_names)}")
    print(f"[INFO] Images found: {len(image_paths)}")

    model = load_recognition_model()

    embeddings, kept_paths, failed_paths = extract_embeddings(model, image_paths)

    kept_set = set(kept_paths)

    filtered_labels = []
    filtered_identities = []
    filtered_paths = []

    for p, lbl, ident in zip(image_paths, labels, identities):
        if p in kept_set:
            filtered_labels.append(lbl)
            filtered_identities.append(ident)
            filtered_paths.append(p)

    filtered_labels = np.array(filtered_labels, dtype=np.int64)
    filtered_identities = np.array(filtered_identities)
    filtered_paths = np.array(filtered_paths)
    class_names = np.array(class_names)

    print(f"[INFO] Successful embeddings: {len(filtered_paths)}")
    print(f"[INFO] Failed images: {len(failed_paths)}")
    print(f"[INFO] Embedding shape: {embeddings.shape}")

    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,          # [N, D]
        labels=filtered_labels,         # [N]
        identities=filtered_identities, # [N]
        paths=filtered_paths,           # [N]
        class_names=class_names,        # [num_classes]
        failed_paths=np.array(failed_paths),
    )

    print(f"[INFO] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
