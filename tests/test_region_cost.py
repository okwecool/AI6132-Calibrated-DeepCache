import unittest

import numpy as np
from PIL import Image
import torch

from src.region_cost import boundary_crops, edit_regions, normalized_region_weights


class RegionCostTests(unittest.TestCase):
    def test_ring_is_added_before_area_projection(self):
        pixels = np.zeros((4, 4), dtype=np.uint8)
        pixels[:2, :2] = 255
        mask = Image.fromarray(pixels)
        edit, ring = edit_regions(mask, radius=1)
        self.assertEqual(int(edit.sum()), 12)
        self.assertEqual(int(ring.sum()), 5)
        self.assertTrue(np.all(edit[ring]))

        weights = normalized_region_weights(mask, (2, 2), radius=1)
        expected = torch.tensor([
            [[1, 1], [1, 1]],
            [[0, 1], [1, 1]],
            [[0, 1.5], [1.5, 1.25]],
            [[0, 2.5], [2.5, 1.75]],
        ])
        expected /= expected.sum(dim=(1, 2), keepdim=True)
        torch.testing.assert_close(weights, expected)

    def test_perimeter_crops_are_unique_and_inside_image(self):
        pixels = np.zeros((512, 512), dtype=np.uint8)
        pixels[7:460, 20:480] = 255
        mask = Image.fromarray(pixels)
        boxes = boundary_crops(mask)
        self.assertEqual(len(set(boxes)), 8)
        self.assertEqual(boxes, boundary_crops(mask))
        for left, top, right, bottom in boxes:
            self.assertTrue(0 <= left < right <= 512 and 0 <= top < bottom <= 512)
            self.assertEqual((right - left, bottom - top), (128, 128))


if __name__ == "__main__":
    unittest.main()
