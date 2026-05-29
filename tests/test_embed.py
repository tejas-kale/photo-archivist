import unittest
from unittest.mock import Mock, patch

import numpy as np


class Tensor:
    def __init__(self, values):
        self.values = np.array(values, dtype="float32")

    def detach(self):
        return self

    def numpy(self):
        return self.values


class Output:
    def __init__(self, values):
        self.pooler_output = Tensor(values)


class EmbedTests(unittest.TestCase):
    def test_embedding_accepts_pooled_model_output(self):
        import embed

        image = Mock()
        image.convert.return_value = image
        model = Mock()
        processor = Mock(return_value={"pixel_values": "pixels"})
        model.get_image_features.return_value = Output([[3, 4]])

        with patch.object(embed, "model", return_value=(model, processor)), patch.object(embed.Image, "open", return_value=image):
            vector = embed.embedding("x.jpg")

        np.testing.assert_allclose(vector, np.array([0.6, 0.8], dtype="float32"))


if __name__ == "__main__":
    unittest.main()
