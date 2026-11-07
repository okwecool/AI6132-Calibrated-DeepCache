"""Spatial weights and fixed evaluation crops for background replacement."""

import numpy as np
from PIL import ImageFilter
import torch
import torch.nn.functional as F


def edit_regions(foreground, radius):
    """Return the editable mask and its band immediately outside the product."""
    protected = np.asarray(foreground) == 255
    edit = ~protected
    ring = (np.asarray(foreground.filter(ImageFilter.MaxFilter(2 * radius + 1))) == 255) & edit
    return edit, ring


def normalized_region_weights(foreground, latent_size, radius=16):
    """Average pixel weights into latent cells before normalizing each objective."""
    edit, ring = edit_regions(foreground, radius)
    # The ring is contained in the editable area, so E * (1 + lambda * R)
    # equals E + lambda * R on the pixel grid.
    edit = edit.astype(np.float32)
    ring = ring.astype(np.float32)
    pixel_weights = np.stack((np.ones_like(edit), edit, edit + ring, edit + 3 * ring))
    weights = F.interpolate(torch.from_numpy(pixel_weights)[None], size=latent_size, mode="area")[0]
    totals = weights.sum(dim=(1, 2), keepdim=True)
    if (totals == 0).any():
        raise ValueError("Expected a nonempty editable region")
    return weights / totals


def boundary_crops(foreground, size=128, count=8):
    """Spread fixed crops across the mask perimeter by farthest-point sampling."""
    protected = np.asarray(foreground) == 255
    eroded = np.asarray(foreground.filter(ImageFilter.MinFilter(3))) == 255
    points = np.argwhere(protected & ~eroded)
    if len(points) == 0:
        raise ValueError("Expected a foreground perimeter")
    starts = np.maximum(points - size // 2, 0)
    starts = np.minimum(starts, np.array(protected.shape) - size)
    starts = np.unique(starts, axis=0)
    nearest = np.full(len(starts), np.inf)
    index = 0
    boxes = []
    for _ in range(min(count, len(starts))):
        y, x = map(int, starts[index])
        boxes.append((x, y, x + size, y + size))
        nearest = np.minimum(nearest, ((starts - (y, x)) ** 2).sum(axis=1))
        index = int(np.argmax(nearest))
    return boxes
