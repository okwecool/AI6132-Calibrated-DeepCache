"""Image and foreground mask handling for product background replacement."""

from PIL import Image, ImageOps


def prepare_product(image, foreground, size=512):
    """Fit aligned inputs to a square canvas. White foreground is protected."""
    if image.size != foreground.size:
        raise ValueError("The image and foreground mask must have matching dimensions")
    image = image.convert("RGB")
    foreground = foreground.convert("L").point(lambda pixel: 255 if pixel >= 128 else 0)
    fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    mask = foreground.resize(fitted.size, Image.Resampling.NEAREST)
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    protection = Image.new("L", canvas.size, 0)
    canvas.paste(fitted, offset)
    protection.paste(mask, offset)
    if protection.getextrema() != (0, 255):
        raise ValueError("The prepared mask must contain both foreground and editable pixels")
    return canvas, protection


def restore_product(source, foreground, generated):
    """Restore the exact prepared source pixels inside the binary foreground."""
    if source.size != generated.size or source.size != foreground.size:
        raise ValueError("Source, mask and generated image must have matching dimensions")
    return Image.composite(source, generated, foreground)
