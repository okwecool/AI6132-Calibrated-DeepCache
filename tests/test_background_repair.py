import unittest

import numpy as np
from PIL import Image

from src.background_repair import composite_repair, hide_product, repair_masks


class BackgroundRepairTests(unittest.TestCase):
    def setUp(self):
        pixels = np.zeros((8, 8), dtype=np.uint8)
        pixels[2:6, 2:5] = 255
        self.foreground = Image.fromarray(pixels)
        self.protected = pixels == 255
        source = np.zeros((8, 8, 3), dtype=np.uint8)
        source[:] = (10, 50, 20)
        source[self.protected] = (255, 0, 0)
        self.source = Image.fromarray(source)

    def test_repair_keeps_foreground_and_outer_pixels(self):
        mask, alpha = repair_masks(self.foreground, [(1, 1), (6, 1), (6, 6), (1, 6)], feather=2)
        editable = np.asarray(mask) == 255
        self.assertFalse(editable[self.protected].any())
        generated = Image.new("RGB", (8, 8), (100, 100, 100))
        result = composite_repair(self.source, generated, self.foreground, mask, alpha)
        changed = np.any(np.asarray(result) != np.asarray(self.source), axis=2)
        np.testing.assert_array_equal(changed, editable)
        self.assertEqual(alpha.getpixel((1, 1)), 128)

    def test_product_hide_uses_background_or_fallback_only_in_condition(self):
        local, _ = repair_masks(self.foreground, [(3, 1), (7, 1), (7, 6), (3, 6)])
        hidden = np.asarray(hide_product(self.source, self.foreground, local, (70, 80, 90)))
        np.testing.assert_array_equal(hidden[~self.protected], np.asarray(self.source)[~self.protected])
        self.assertTrue(np.all(hidden[self.protected] == (10, 50, 20)))

        full, _ = repair_masks(self.foreground, [(0, 0), (7, 0), (7, 7), (0, 7)])
        hidden = np.asarray(hide_product(self.source, self.foreground, full, (70, 80, 90)))
        self.assertTrue(np.all(hidden[self.protected] == (70, 80, 90)))


if __name__ == "__main__":
    unittest.main()
