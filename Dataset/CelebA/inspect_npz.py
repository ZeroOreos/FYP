# python3 inspect_npz.py -> NPZ key/value summary
import numpy as np

d = np.load("InsightFace/main_antelopev2_embeddings.npz")


print(d.files)
print(d["embeddings"].shape)
print(d["labels"][:10])
