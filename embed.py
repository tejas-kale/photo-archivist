import argparse
import base64
import subprocess
import sys
from functools import cache

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener


@cache
def model():
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name), CLIPProcessor.from_pretrained(name)


def embedding(path):
    if str(path).lower().endswith((".heic", ".heif")):
        register_heif_opener()
    m, p = model()
    inputs = p(images=Image.open(path).convert("RGB"), return_tensors="pt")
    features = m.get_image_features(**inputs)
    features = getattr(features, "pooler_output", features)
    vector = features.detach().numpy()[0]
    vector = vector / np.linalg.norm(vector)
    return vector.astype("float32")


def embedding_blob(path):
    return embedding(path).tobytes()


def embedding_blob_subprocess(path):
    r = subprocess.run([sys.executable, "-m", "embed", str(path)], check=True, capture_output=True, text=True)
    return base64.b64decode(r.stdout.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    args = p.parse_args()
    print(base64.b64encode(embedding_blob(args.image)).decode())


if __name__ == "__main__":
    main()
