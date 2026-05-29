from functools import cache

import numpy as np
from PIL import Image


@cache
def model():
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name), CLIPProcessor.from_pretrained(name)


def embedding(path):
    m, p = model()
    inputs = p(images=Image.open(path).convert("RGB"), return_tensors="pt")
    features = m.get_image_features(**inputs)
    features = getattr(features, "pooler_output", features)
    vector = features.detach().numpy()[0]
    vector = vector / np.linalg.norm(vector)
    return vector.astype("float32")


def embedding_blob(path):
    return embedding(path).tobytes()
