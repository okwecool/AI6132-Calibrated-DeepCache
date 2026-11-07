"""Constrain a second inpainting pass to an annotated background region."""

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


def repair_masks(foreground, polygon, feather=8):
    """Clip an annotated polygon to the background and feather its outer edge."""
    if feather < 0:
        raise ValueError("Feather width must be nonnegative")
    annotation = Image.new("L", foreground.size, 0)
    ImageDraw.Draw(annotation).polygon([tuple(point) for point in polygon], fill=255)
    marked = np.asarray(annotation) == 255
    protected = np.asarray(foreground) >= 128
    editable = marked & ~protected
    if not editable.any():
        raise ValueError("The repair region contains no editable background")
    if feather and not marked.all():
        # Feather the annotation's edge. The product boundary stays protected
        # by the final intersection, including on thin or hollow silhouettes.
        opacity = np.minimum(distance_transform_edt(marked) / feather, 1.0)
    else:
        opacity = np.ones(marked.shape)
    alpha = np.rint(255 * opacity * editable).astype(np.uint8)
    return Image.fromarray(editable.astype(np.uint8) * 255), Image.fromarray(alpha)


def hide_product(source, foreground, repair_mask, fallback_rgb):
    """Fill product pixels with background sampled outside the editable patch."""
    protected = np.asarray(foreground) >= 128
    repair = np.asarray(repair_mask) >= 128
    pixels = np.asarray(source.convert("RGB")).copy()
    support = ~(protected | repair)
    if support.any():
        nearest = distance_transform_edt(~support, return_distances=False, return_indices=True)
        pixels[protected] = pixels[nearest[0][protected], nearest[1][protected]]
    else:
        pixels[protected] = fallback_rgb
    return Image.fromarray(pixels)


def composite_repair(source, generated, foreground, repair_mask, alpha):
    """Blend the generated patch and enforce exact foreground and exterior pixels."""
    if not source.size == generated.size == foreground.size == repair_mask.size == alpha.size:
        raise ValueError("Repair inputs must use the same canvas")
    protected = np.asarray(foreground) >= 128
    editable = np.asarray(repair_mask) >= 128
    if np.any(protected & editable) or np.any(np.asarray(alpha)[~editable]):
        raise ValueError("Repair masks must exclude the product and the unmarked background")
    result = Image.composite(generated, source, alpha)
    if not np.array_equal(np.asarray(result)[~editable], np.asarray(source)[~editable]):
        raise RuntimeError("The composite modified protected or unmarked pixels")
    return result
