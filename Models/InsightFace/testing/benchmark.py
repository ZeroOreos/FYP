# python3 benchmark.py -> benchmark output
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import time
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

IMAGE_DIR = "/Users/jeromeharianto/Documents/School2/FYP/Images"

TESTS = [
    {
        "label": "buffalo_l_cpu",
        "model": "buffalo_l",
        "providers": ["CPUExecutionProvider"],
        "ctx_id": -1,
    },
    {
        "label": "buffalo_l_coreml_cpu",
        "model": "buffalo_l",
        "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "ctx_id": 0,
    },
    {
        "label": "antelopev2_cpu",
        "model": "antelopev2",
        "providers": ["CPUExecutionProvider"],
        "ctx_id": -1,
    },
    {
        "label": "antelopev2_coreml_cpu",
        "model": "antelopev2",
        "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "ctx_id": 0,
    },
]

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
            paths.append(p)
    return paths

def print_runtime_info():
    print("onnxruntime:", ort.__version__)
    print("available providers:", ort.get_available_providers())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("OMP_WAIT_POLICY:", os.environ.get("OMP_WAIT_POLICY"))

def benchmark_one(test, image_paths):
    print(f"\n=== {test['label']} ===")
    app = FaceAnalysis(name=test["model"], providers=test["providers"])
    app.prepare(ctx_id=test["ctx_id"], det_size=(640, 640))

    warm = cv2.imread(image_paths[0])
    _ = app.get(warm)

    total_time = 0.0
    total_faces = 0
    count = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print("skip unreadable:", path)
            continue

        t0 = time.perf_counter()
        faces = app.get(img)
        dt = time.perf_counter() - t0

        total_time += dt
        total_faces += len(faces)
        count += 1

    if count == 0:
        print("no readable images")
        return

    print("images:", count)
    print("faces found:", total_faces)
    print("total time: %.3fs" % total_time)
    print("avg per image: %.2f ms" % ((total_time / count) * 1000))

def main():
    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        raise RuntimeError("No images found in IMAGE_DIR")

    print_runtime_info()

    for test in TESTS:
        try:
            benchmark_one(test, image_paths)
        except Exception as e:
            print(f"\n=== {test['label']} FAILED ===")
            print(repr(e))

if __name__ == "__main__":
    main()
