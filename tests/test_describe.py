import json
import unittest

from photo_archivist.describe import DEFAULT_PROMPT, VisionResult, parse


class DescribeTests(unittest.TestCase):
    def test_default_prompt_requests_rich_description_prose(self):
        self.assertIn("description_prose", DEFAULT_PROMPT)
        self.assertIn("detailed sentences", DEFAULT_PROMPT)
        self.assertIn("clothing", DEFAULT_PROMPT)
        self.assertIn("setting", DEFAULT_PROMPT)
        self.assertIn("mood", DEFAULT_PROMPT)
        self.assertIn("Do not identify people by name", DEFAULT_PROMPT)
        self.assertIn("Avoid private addresses", DEFAULT_PROMPT)

    def test_parse_preserves_long_description_prose(self):
        text = json.dumps({
            "rating": "keep",
            "cull_reason": "",
            "focus": "sharp",
            "exposure": "adequate",
            "depth_of_field": "standard",
            "noise": "clean",
            "lighting": "indoor artificial",
            "time_of_day": "daytime",
            "dominant_color_palette": "warm",
            "dominant_colors": ["red", "green"],
            "people_count": 5,
            "keywords": ["family", "celebration"],
            "description_prose": "Five people are posing together in what appears to be an indoor celebration. The main woman is wearing a vibrant pink and green saree with gold embroidery, smiling warmly at the camera. Two young girls are seated close to her, dressed in complementary traditional clothing.",
            "activity": "posing smiling",
        })
        result = parse(text)
        self.assertEqual("keep", result.rating)
        self.assertEqual(5, result.people_count)
        self.assertIn("saree with gold embroidery", result.description_prose)
        self.assertEqual(["family", "celebration"], result.keywords)

    def test_parse_falls_back_to_description_alias(self):
        text = json.dumps({"description": "A dog in a park.", "rating": "keep"})
        result = parse(text)
        self.assertEqual("A dog in a park.", result.description_prose)

    def test_vision_result_get_uses_aliases(self):
        result = VisionResult(description_prose="test", people_count=3, time_of_day="day", lighting="bright")
        self.assertEqual("test", result.get("description"))
        self.assertEqual(3, result.get("number_people"))
        self.assertEqual("day", result.get("day_night"))
        self.assertEqual("bright", result.get("lighting_quality"))

    def test_vision_result_getitem_uses_aliases(self):
        result = VisionResult(description_prose="test", people_count=3)
        self.assertEqual("test", result["description"])
        self.assertEqual(3, result["number_people"])


if __name__ == "__main__":
    unittest.main()
