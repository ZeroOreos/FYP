# python3 test_insightFace.py -> smoke-test output
from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)  # CPU

img = cv2.imread("Images/test0.jpeg")
faces = app.get(img)

print(len(faces))
print(faces[0].embedding.shape)
